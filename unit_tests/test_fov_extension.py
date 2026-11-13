"""`RandomFOVExtensionGPU`: shift the scan inside the patch, fill what lies outside it.

The fixture is a small sagittal-slab stand-in: tissue in [1, 2], a slab of exact zeros at
both ends of the first axis (nnU-Net's padding), and a labelled block at intensity 1000.
The sentinel makes two properties checkable voxel by voxel: the image and the mask moved
together (1000 sits exactly under the label), and no foreground intensity reached the
synthesised background (nothing else comes near 1000).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.context import RandomFOVExtensionGPU, _dilate, _em_regions, _extend_regions, _shift, _touched_faces
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (3, 1, 28, 24, 20)  # non-cubic on purpose
SENTINEL = 1000.0


def slab(shape=SHAPE):
    image = torch.rand(shape) + 1.0
    image[:, :, :6] = 0.0
    image[:, :, -6:] = 0.0
    seg = torch.zeros(shape)
    seg[:, :, 10:18, 6:16, 7:13] = 1.0
    image[seg.bool()] = SENTINEL
    return image, seg


def through(shape=SHAPE, h=slice(None)):
    """The slab with its label running along the second axis, through both faces by default."""
    image, seg = slab(shape)
    image[seg.bool()] = torch.rand_like(image)[seg.bool()] + 1.0
    seg.zero_()
    seg[:, :, 10:18, h, 7:13] = 1.0
    image[seg.bool()] = SENTINEL
    return image, seg


def bands(axis, levels, shape=SHAPE):
    """No padding and no label: equal bands of constant tissue along `axis`, at `levels`."""
    image = torch.empty(shape)
    for i, chunk in enumerate(torch.arange(shape[2 + axis]).tensor_split(len(levels))):
        image.index_fill_(2 + axis, chunk, levels[i])
    return image + 0.05 * torch.rand(shape), torch.zeros(shape)


def with_shift(transform, image, seg, shift):
    params = transform.generate_parameters(image.shape)
    params["shift"] = torch.tensor(shift, dtype=torch.float32).expand(image.shape[0], 3).clone()
    params["seg"] = seg
    return params


def run(transform, image, seg, same_on_batch=True):
    pipeline = AugmentationSequentialCustom(transform, data_keys=["input", "mask"], same_on_batch=same_on_batch)
    return pipeline(image.clone(), seg.clone())


class TestShiftAndFill(SmaugLabTestCase):
    def test_image_and_mask_move_together(self):
        image, seg = slab()
        for seed in range(6):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                out, out_seg = run(RandomFOVExtensionGPU(shift=1.0, p=1.0), image, seg)
                self.assertTrue(bool(torch.equal(out > SENTINEL / 2, out_seg > 0)))

    def test_the_shift_moves_something(self):
        """The guard against passing the alignment test by never shifting."""
        image, seg = slab()
        torch.manual_seed(0)
        _, out_seg = run(RandomFOVExtensionGPU(shift=1.0, p=1.0), image, seg)
        self.assertFalse(bool(torch.equal(out_seg, seg)))

    def test_outside_the_fill_is_the_shifted_input(self):
        image, seg = slab()
        transform = RandomFOVExtensionGPU(shift=0.5, p=1.0)
        torch.manual_seed(3)
        params = transform.generate_parameters(image.shape)
        params["seg"] = seg
        out = transform.apply_transform(image, params, transform.flags)
        offsets = transform._offsets(params, image.shape[2:])
        shifted = _shift(image, offsets, 0.0)
        scan = _shift(torch.ones_like(image, dtype=torch.bool), offsets, False) & (shifted != 0.0)
        self.assertTrue(bool(torch.equal(out[scan], shifted[scan])))

    def test_no_shift_touches_only_the_padding(self):
        image, seg = slab()
        out, out_seg = run(RandomFOVExtensionGPU(shift=0.0, p=1.0), image, seg)
        scan = image != 0.0
        self.assertTrue(bool(torch.equal(out_seg, seg)))
        self.assertTrue(bool(torch.equal(out[scan], image[scan])))
        self.assertEqual(int((out[~scan] == 0.0).sum()), 0, "padding left unfilled")

    def test_no_foreground_intensity_reaches_the_fill(self):
        image, seg = slab()
        for margin in (0, 2):
            with self.subTest(margin=margin):
                torch.manual_seed(1)
                out, out_seg = run(RandomFOVExtensionGPU(shift=1.0, margin=margin, p=1.0), image, seg)
                background = out[out_seg == 0]
                self.assertLess(float(background.max()), 10.0)
                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_unselected_rows_are_untouched(self):
        image, seg = slab()
        torch.manual_seed(0)
        out, out_seg = run(RandomFOVExtensionGPU(p=0.0), image, seg)
        self.assertTrue(bool(torch.equal(out, image)))
        self.assertTrue(bool(torch.equal(out_seg, seg)))

    def test_no_background_still_shifts_consistently(self):
        """Nothing to fit: the fill stays at pad_value, and the mask still follows the image."""
        image, seg = slab()
        seg[:, :, 6:-6] = 1.0
        image[seg.bool()] = SENTINEL
        torch.manual_seed(2)
        # Not spared: the label touches both faces of two axes, which would pin them.
        out, out_seg = run(RandomFOVExtensionGPU(shift=0.5, spare_label_faces=False, p=1.0), image, seg)
        self.assertTrue(bool(torch.equal(out > SENTINEL / 2, out_seg > 0)))
        self.assertTrue(bool(((out == 0.0) | (out == SENTINEL)).all()))

    def test_without_a_segmentation_every_scan_voxel_is_a_source(self):
        image, _ = slab()
        transform = RandomFOVExtensionGPU(shift=0.0, p=1.0)
        params = transform.generate_parameters(image.shape)
        out = transform.apply_transform(image, params, transform.flags)
        self.assertEqual(int((out == 0.0).sum()), 0)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_padding_means_every_channel(self):
        image, seg = slab((2, 2, 28, 24, 20))
        image[:, 1, :6] = 5.0  # channel 1 is not padding there, so neither is the voxel
        out, _ = run(RandomFOVExtensionGPU(shift=0.0, p=1.0), image, seg)
        self.assertTrue(bool(torch.equal(out[:, :, :6], image[:, :, :6])))
        self.assertEqual(int((out[:, :, -6:] == 0.0).all(dim=1).sum()), 0)


class TestFillFromTheWholePatch(SmaugLabTestCase):
    """The region map is built before the shift, so a large shift does not narrow the fill."""

    def test_fill_uses_background_the_shift_removed(self):
        image, seg = bands(0, (1.0, 7.0))
        transform = RandomFOVExtensionGPU(regions=(2, 2), p=1.0)
        for seed in range(4):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                params = with_shift(transform, image, seg, [0.5, 0.0, 0.0])  # the 7-band leaves the window
                out = transform.apply_transform(image, params, transform.flags)
                fill = out[:, :, :14]
                self.assertGreater(float((fill > 4.0).float().mean()), 0.2)

    def test_large_shift_fill_is_not_one_region(self):
        image, seg = bands(2, (1.0, 4.0, 7.0))
        transform = RandomFOVExtensionGPU(regions=(3, 3), p=1.0)
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                params = with_shift(transform, image, seg, [-0.6, 0.5, 0.5])  # ~11% of the scan stays
                out = transform.apply_transform(image, params, transform.flags)
                fill = ~_shift(torch.ones_like(image, dtype=torch.bool), transform._offsets(params, image.shape[2:]), False)
                level = torch.bucketize(out[fill], torch.tensor([2.5, 5.5]))
                share = level.bincount(minlength=3).float() / level.numel()
                self.assertLess(float(share.max()), 0.7, share.tolist())
                self.assertGreater(float(share.min()), 0.1, share.tolist())

    def test_layers_cover_the_fill(self):
        torch.manual_seed(0)
        source = torch.arange(3).repeat_interleave(8).view(1, 1, 1, 24, 1).expand(8, 1, 28, 24, 20).clone()
        source[..., :10] = -1  # half the source is unlabelled, as the spine and the padding are
        seed = torch.full_like(source, -1)
        layout = _extend_regions(source, seed, torch.ones_like(source, dtype=torch.bool), layers=6, close_iters=0, default=-1)
        self.assertLess(float((layout == -1).float().mean()), 0.1)
        share = layout[layout >= 0].bincount(minlength=3).float() / int((layout >= 0).sum())
        self.assertLess(float(share.max()), 0.5, share.tolist())  # none takes over
        self.assertGreater(float(share.min()), 0.15, share.tolist())

    def test_a_vacated_window_gets_a_varied_fill(self):
        image, seg = slab()
        transform = RandomFOVExtensionGPU(p=1.0)
        torch.manual_seed(4)
        params = with_shift(transform, image, seg, [1.0, 0.0, 0.0])  # all 28 voxels of the first axis
        out = transform.apply_transform(image, params, transform.flags)
        out_seg = transform.apply_transform_mask(seg, params, transform.flags)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertEqual(int((out == 0.0).sum()), 0)
        self.assertTrue(bool((out.flatten(1).std(dim=1) > 0.05).all()))
        self.assertEqual(int(out_seg.abs().sum()), 0)
        self.assertLess(float(out.max()), 10.0)


class TestSparedFaces(SmaugLabTestCase):
    def test_a_structure_through_the_window_is_not_cut(self):
        image, seg = through()
        moved = []
        for seed in range(6):
            torch.manual_seed(seed)
            _, spared = run(RandomFOVExtensionGPU(shift=(0.0, 1.0, 0.0), p=1.0), image, seg)
            self.assertTrue(bool(torch.equal(spared, seg)), seed)
            torch.manual_seed(seed)
            _, cut = run(RandomFOVExtensionGPU(shift=(0.0, 1.0, 0.0), spare_label_faces=False, p=1.0), image, seg)
            moved.append(not bool(torch.equal(cut, seg)))
        self.assertTrue(any(moved), "the unspared twin never shifted")

    def test_a_structure_touching_one_face_leaves_only_through_it(self):
        image, seg = through(h=slice(0, 12))  # touches the low face of the second axis only
        transform = RandomFOVExtensionGPU(shift=(0.0, 1.0, 0.0), p=1.0)
        offsets = []
        for seed in range(8):
            torch.manual_seed(seed)
            params = transform.generate_parameters(image.shape)
            params["seg"] = seg
            offsets.append(transform._offsets(params, image.shape[2:])[:, 1])
            torch.manual_seed(seed)
            out, out_seg = run(transform, image, seg)
            self.assertTrue(bool(torch.equal(out > SENTINEL / 2, out_seg > 0)))
        offsets = torch.cat(offsets)
        self.assertTrue(bool((offsets <= 0).all()), offsets.tolist())
        self.assertTrue(bool((offsets < 0).any()), offsets.tolist())

    def test_touch_tolerance_is_margin(self):
        labelled = torch.zeros(1, 1, 8, 8, 20, dtype=torch.bool)
        labelled[..., 3:17] = True  # 3 empty voxels at each face of the last axis
        self.assertTrue(bool(_touched_faces(labelled, 4)[0, 2].all()))
        self.assertFalse(bool(_touched_faces(labelled, 2)[0, 2].any()))
        self.assertTrue(bool(_touched_faces(labelled, 2)[0, :2].all()), "the first two axes are labelled wall to wall")

    def test_sparing_is_per_row_and_moves_image_and_mask_together(self):
        shape = (4, 1, 28, 24, 20)
        image, seg = slab(shape)
        t_image, t_seg = through(shape)
        image[::2], seg[::2] = t_image[::2], t_seg[::2]  # rows 0 and 2 run through the second axis
        transform = RandomFOVExtensionGPU(shift=1.0, p=0.5)
        others = []
        for seed in range(8):
            torch.manual_seed(seed)
            out, out_seg = run(transform, image, seg, same_on_batch=True)
            self.assertTrue(bool(torch.equal(out > SENTINEL / 2, out_seg > 0)))
            params = transform.generate_parameters(shape)
            params["seg"] = seg
            offsets = transform._offsets(params, shape[2:])
            self.assertTrue(bool((offsets[::2, 1] == 0).all()))
            others.append(offsets[1::2, 1])
        self.assertTrue(bool((torch.cat(others) != 0).any()))

    def test_mask_hook_reads_params_seg(self):
        image, seg = slab()
        _, spine = through()
        transform = RandomFOVExtensionGPU(shift=1.0, p=1.0)
        torch.manual_seed(0)
        params = transform.generate_parameters(image.shape)
        params["seg"] = spine
        out_seg = transform.apply_transform_mask(seg, params, transform.flags)
        offsets = transform._offsets(params, image.shape[2:])
        self.assertTrue(bool((offsets[:, 1] == 0).all()))
        self.assertTrue(bool(torch.equal(out_seg, _shift(seg, offsets, 0))))

    def test_sparing_consumes_no_randomness(self):
        image, seg = slab()  # touches no face, so sparing must change nothing at all
        results, states = [], []
        for spare in (True, False):
            torch.manual_seed(9)
            results.append(run(RandomFOVExtensionGPU(shift=1.0, spare_label_faces=spare, p=1.0), image, seg))
            states.append(torch.get_rng_state())
        self.assertTrue(bool(torch.equal(states[0], states[1])))
        self.assertTrue(bool(torch.equal(results[0][0], results[1][0])))
        self.assertTrue(bool(torch.equal(results[0][1], results[1][1])))


class TestSelectionAndDeterminism(SmaugLabTestCase):
    def test_rows_are_selected_per_sample_under_same_on_batch(self):
        """The paper config hands every child same_on_batch=True; this transform ignores it."""
        image, seg = slab((8, 1, 28, 24, 20))
        counts = []
        for seed in range(8):
            torch.manual_seed(seed)
            transform = RandomFOVExtensionGPU(shift=0.0, p=0.5)
            out, _ = run(transform, image, seg, same_on_batch=True)
            self.assertFalse(transform.same_on_batch)
            counts.append(int((out != image).flatten(1).any(dim=1).sum()))
        self.assertTrue(any(0 < c < 8 for c in counts), counts)

    def test_seed_determinism(self):
        image, seg = slab()
        results = []
        for _ in range(2):
            torch.manual_seed(11)
            results.append(run(RandomFOVExtensionGPU(shift=0.5, p=1.0), image, seg))
        self.assertTrue(bool(torch.equal(results[0][0], results[1][0])))
        self.assertTrue(bool(torch.equal(results[0][1], results[1][1])))

    def test_autocast_matches_fp32(self):
        for name, (image, seg) in (("slab", slab()), ("through", through())):
            with self.subTest(fixture=name):
                torch.manual_seed(5)
                ref, ref_seg = run(RandomFOVExtensionGPU(shift=0.5, p=1.0), image, seg)
                torch.manual_seed(5)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out, out_seg = run(RandomFOVExtensionGPU(shift=0.5, p=1.0), image, seg)
                self.assertTrue(bool(torch.equal(out_seg, ref_seg)))
                self.assertTrue(bool(torch.allclose(out.float(), ref, atol=1e-5)))

    def test_rejects_bad_arguments(self):
        for kwargs in ({"shift": 1.5}, {"shift": (0.5, 0.5)}, {"regions": (0, 3)}, {"regions": (5, 3)}, {"layers": 0}, {"margin": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                RandomFOVExtensionGPU(**kwargs)
        with self.assertRaises(TypeError):
            RandomFOVExtensionGPU(spare_label_faces="false")


class TestHelpers(SmaugLabTestCase):
    def test_em_separates_two_populations(self):
        x = torch.cat([torch.randn(1, 4000) * 0.1, torch.randn(1, 4000) * 0.1 + 5.0], dim=1)
        assign = _em_regions(x, torch.ones_like(x), torch.tensor([2]), k_max=4)
        first, second = assign[0, :4000], assign[0, 4000:]
        self.assertGreater(float((first == first.mode().values).float().mean()), 0.99)
        self.assertGreater(float((second == second.mode().values).float().mean()), 0.99)
        self.assertNotEqual(int(first.mode().values), int(second.mode().values))
        self.assertLess(int(assign.max()), 2, "a masked-out component was assigned")

    def test_em_fits_only_weighted_voxels(self):
        """With weight on one population only, both components land inside it."""
        x = torch.cat([torch.randn(1, 4000) * 0.1, torch.randn(1, 4000) * 0.1 + 5.0], dim=1)
        weight = torch.cat([torch.ones(1, 4000), torch.zeros(1, 4000)], dim=1)
        assign = _em_regions(x, weight, torch.tensor([2]), k_max=2)
        self.assertEqual(int(assign[0, 4000:].unique().numel()), 1)

    def test_dilate_matches_a_dense_pool(self):
        mask = torch.rand(2, 1, 12, 10, 9) > 0.95
        for radius in (0, 1, 3):
            with self.subTest(radius=radius):
                k = 2 * radius + 1
                dense = F.max_pool3d(mask.float(), k, stride=1, padding=radius) > 0
                self.assertTrue(bool(torch.equal(_dilate(mask, radius), dense)))

    def test_shift_matches_slicing(self):
        volume = torch.rand(5, 2, 7, 6, 5)
        offsets = [[0, 0, 0], [3, -2, 1], [-7, 0, 0], [1, 6, -4], [-2, -5, 4]]
        expected = torch.full_like(volume, -3.0)
        for b, offset in enumerate(offsets):
            src, dst = [], []
            for size, o in zip(volume.shape[2:], offset):
                src.append(slice(max(0, -o), max(0, size - max(0, o))))
                dst.append(slice(max(0, o), max(0, size - max(0, -o))))
            if all(s.stop > s.start for s in src):
                expected[b][(slice(None), *dst)] = volume[b][(slice(None), *src)]
        self.assertTrue(bool(torch.equal(_shift(volume, offsets, -3.0), expected)))
        self.assertTrue(bool(torch.equal(_shift(volume, torch.tensor(offsets), -3.0), expected)))

    def test_shift_beyond_the_patch_empties_the_row(self):
        volume = torch.ones(1, 1, 4, 4, 4)
        self.assertTrue(bool((_shift(volume, [[0, 4, 0]], 0.0) == 0).all()))
        self.assertTrue(bool(torch.equal(_shift(volume, [[0, 0, 0]], 0.0), volume)))
