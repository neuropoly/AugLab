"""`RandomFOVWindowGPU`, the GPU half: fill what lies outside the case, on the rows the loader moved.

The fixture is a window that left the case on the first axis: six slices of exact zeros
(nnU-Net's padding), four of a constant -0.5 (a resampled MRI's black border), tissue in
[1, 2], and a labelled block at intensity 1000. The sentinel makes "no foreground intensity
reached the fill" checkable voxel by voxel. The loader half is in `test_fov_window_loader.py`.
"""

from __future__ import annotations

import warnings

import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.context import _SEAM, RandomFOVWindowGPU, WindowSettings, _black_border, _dilate
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (3, 1, 28, 24, 20)
SENTINEL = 1000.0
BLACK = -0.5


def case(shape=SHAPE):
    image = torch.rand(shape) + 1.0
    image[:, :, :6] = 0.0
    image[:, :, 6:10] = BLACK
    seg = torch.zeros(shape)
    seg[:, :, 15:22, 6:16, 7:13] = 1.0
    image[seg.bool()] = SENTINEL
    return image, seg


def bands(levels, shape=SHAPE):
    """Padding on the first six slices, then equal bands of constant tissue along the last axis; no label."""
    image = torch.empty(shape)
    for i, chunk in enumerate(torch.arange(shape[4]).tensor_split(len(levels))):
        image.index_fill_(4, chunk, levels[i])
    image = image + 0.05 * torch.rand(shape)
    image[:, :, :6] = 0.0
    return image, torch.zeros(shape)


def run(transform, image, seg, rows=None, same_on_batch=True):
    pipeline = AugmentationSequentialCustom(transform, data_keys=["input", "mask"], same_on_batch=same_on_batch)
    if rows is not None:
        transform.set_selection(rows)
    return pipeline(image.clone(), seg.clone())


def expected_fill(image, seg, margin, border):
    padding = (image == 0.0).all(dim=1, keepdim=True)
    outside = padding | ((image == BLACK).all(dim=1, keepdim=True) if border else torch.zeros_like(padding))
    return _dilate(outside, _SEAM) & ~_dilate(seg > 0, margin)


class TestSelection(SmaugLabTestCase):
    def test_only_the_loaders_rows_change(self):
        image, seg = case()
        for rows in ([True, False, True], [False, True, False]):
            with self.subTest(rows=rows):
                transform = RandomFOVWindowGPU(p=0.5)
                out, out_seg = run(transform, image, seg, rows)
                changed = (out != image).flatten(1).any(dim=1).tolist()
                self.assertEqual(changed, rows)
                self.assertTrue(bool(torch.equal(out_seg, seg)))
                self.assertEqual(transform.fallbacks, 0)

    def test_the_loaders_rows_hold_on_kornias_own_path(self):
        """Under autocast the container falls back to kornia's forward, which asks `_p_gen` too."""
        image, seg = case()
        transform = RandomFOVWindowGPU(p=0.5)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out, _ = run(transform, image, seg, [False, True, True])
        self.assertEqual((out.float() != image).flatten(1).any(dim=1).tolist(), [False, True, True])

    def test_a_selection_is_used_once_then_it_falls_back_and_warns(self):
        image, seg = case()
        transform = RandomFOVWindowGPU(p=0.5)
        run(transform, image, seg, [True, True, True])
        with self.assertWarns(UserWarning):
            run(transform, image, seg)
        self.assertEqual(transform.fallbacks, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run(transform, image, seg)  # warned once per instance
        self.assertEqual(transform.fallbacks, 2)

    def test_selection_errors(self):
        image, seg = case()
        with self.assertRaises(ValueError):
            run(RandomFOVWindowGPU(p=0.5), image, seg, [True, False])
        with self.assertRaises(ValueError):
            RandomFOVWindowGPU(p=1.0).set_selection([True, False, True])
        with self.assertRaises(ValueError):
            RandomFOVWindowGPU(p=0.0).set_selection([False, True, False])
        transform = RandomFOVWindowGPU(p=0.5)
        transform.set_selection([True, False, True])
        with self.assertRaises(RuntimeError):
            transform.set_selection([True, False, True])

    def test_p_zero_draws_nothing_and_changes_nothing(self):
        image, seg = case()
        torch.manual_seed(3)
        out, out_seg = run(RandomFOVWindowGPU(p=0.0), image, seg, [False, False, False])
        after = torch.get_rng_state()
        torch.manual_seed(3)
        self.assertTrue(bool(torch.equal(after, torch.get_rng_state())))
        self.assertTrue(bool(torch.equal(out, image)))
        self.assertTrue(bool(torch.equal(out_seg, seg)))

    def test_no_selected_row_draws_nothing(self):
        image, seg = case()
        torch.manual_seed(3)
        run(RandomFOVWindowGPU(p=0.5), image, seg, [False, False, False])
        after = torch.get_rng_state()
        torch.manual_seed(3)
        self.assertTrue(bool(torch.equal(after, torch.get_rng_state())))

    def test_loader_settings(self):
        transform = RandomFOVWindowGPU(extend=(0.5, 0.25, 0.0), margin=3, p=0.3)
        self.assertEqual(transform.loader_settings(), WindowSettings(p=0.3, extend=(0.5, 0.25, 0.0), margin=3))
        self.assertEqual(RandomFOVWindowGPU(extend=0.6).loader_settings().extend, (0.6, 0.6, 0.6))

    def test_same_on_batch_is_ignored(self):
        transform = RandomFOVWindowGPU()
        transform.same_on_batch = True
        self.assertFalse(transform.same_on_batch)


class TestFillRegion(SmaugLabTestCase):
    def test_exactly_the_fill_changes(self):
        image, seg = case()
        for kind in ("em", "noise", "voronoi"):
            for border_p in (0.0, 1.0):
                with self.subTest(kind=kind, border_p=border_p):
                    torch.manual_seed(1)
                    out, out_seg = run(RandomFOVWindowGPU(fill=kind, border_p=border_p, p=1.0), image, seg)
                    fill = expected_fill(image, seg, 4, border_p == 1.0)
                    self.assertTrue(bool(torch.equal(out[~fill], image[~fill])))
                    self.assertEqual(int((out[fill] == 0.0).sum()), 0, "padding left in the fill")
                    self.assertEqual(int((out[fill] == BLACK).sum()), 0, "border left in the fill")
                    self.assertTrue(bool(torch.equal(out_seg, seg)))

    def test_nothing_within_margin_of_a_label_changes(self):
        image, seg = case()
        seg[:, :, 3:6] = 1.0  # a label cut by the case face, on the padding
        for margin in (0, 2, 4):
            with self.subTest(margin=margin):
                torch.manual_seed(2)
                out, _ = run(RandomFOVWindowGPU(margin=margin, border_p=1.0, p=1.0), image, seg)
                near = _dilate(seg > 0, margin)
                self.assertTrue(bool(torch.equal(out[near], image[near])))

    def test_no_foreground_intensity_reaches_the_fill(self):
        image, seg = case()
        for kind in ("em", "noise", "voronoi"):
            with self.subTest(kind=kind):
                torch.manual_seed(1)
                out, out_seg = run(RandomFOVWindowGPU(fill=kind, margin=0, border_p=1.0, p=1.0), image, seg)
                self.assertLess(float(out[out_seg == 0].max()), 10.0)
                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_every_kind_is_textured(self):
        image, seg = case()
        for kind in ("em", "noise", "voronoi"):
            with self.subTest(kind=kind):
                torch.manual_seed(4)
                out, _ = run(RandomFOVWindowGPU(fill=kind, p=1.0), image, seg)
                fill = out[:, :, :5]
                self.assertTrue(bool((fill.flatten(1).std(dim=1) > 0.05).all()))

    def test_noise_has_the_backgrounds_moments(self):
        image, seg = case()
        image[:, :, 6:10] = torch.rand_like(image[:, :, 6:10]) + 1.0  # no border: the background is the tissue
        torch.manual_seed(5)
        out, _ = run(RandomFOVWindowGPU(fill="noise", p=1.0), image, seg)
        source = image[:, :, 12:].flatten()
        source = source[source < 10.0]
        fill = out[:, :, :5].flatten()
        self.assertAlmostEqual(float(fill.mean()), float(source.mean()), delta=0.05)
        self.assertGreater(float(fill.std()), 0.4 * float(source.std()))
        self.assertLess(float(fill.std()), 1.5 * float(source.std()))

    def test_voronoi_cells_take_several_regions(self):
        image, seg = bands((1.0, 4.0, 7.0))
        shares = []
        for seed in range(6):
            torch.manual_seed(seed)
            out, _ = run(RandomFOVWindowGPU(fill="voronoi", regions=(3, 3), cells=(8, 8), p=1.0), image, seg)
            level = torch.bucketize(out[:, :, :5].flatten(), torch.tensor([2.5, 5.5]))
            shares.append(level.bincount(minlength=3).float() / level.numel())
        self.assertGreater(float(torch.stack(shares).min(dim=0).values.max()), 0.0)
        self.assertTrue(all(int((s > 0.05).sum()) >= 2 for s in shares), [s.tolist() for s in shares])

    def test_a_list_draws_every_kind(self):
        transform = RandomFOVWindowGPU(fill=["em", "noise", "voronoi"], p=1.0)
        torch.manual_seed(0)
        drawn = torch.cat([transform.generate_parameters(SHAPE)["fill"] for _ in range(20)])
        self.assertEqual(sorted(drawn.unique().tolist()), [0, 1, 2])

    def test_a_row_without_background_is_left_alone(self):
        image, seg = case()
        seg[:, :, 6:] = 1.0
        image[seg.bool()] = SENTINEL
        out, _ = run(RandomFOVWindowGPU(p=1.0), image, seg)
        self.assertTrue(bool(torch.equal(out, image)))

    def test_without_a_segmentation(self):
        image, _ = case()
        transform = RandomFOVWindowGPU(p=1.0, border_p=1.0)
        params = transform.generate_parameters(image.shape)
        out = transform.apply_transform(image, params, transform.flags)
        self.assertEqual(int((out[:, :, :10] == 0.0).sum()), 0)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_seed_determinism(self):
        image, seg = case()
        results = []
        for _ in range(2):
            torch.manual_seed(11)
            results.append(run(RandomFOVWindowGPU(fill=["em", "noise", "voronoi"], border_p=0.5, p=1.0), image, seg)[0])
        self.assertTrue(bool(torch.equal(results[0], results[1])))

    def test_autocast_matches_fp32(self):
        image, seg = case()
        for kind in ("em", "noise", "voronoi"):
            with self.subTest(kind=kind):
                torch.manual_seed(5)
                ref, _ = run(RandomFOVWindowGPU(fill=kind, border_p=1.0, p=1.0), image, seg)
                torch.manual_seed(5)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out, _ = run(RandomFOVWindowGPU(fill=kind, border_p=1.0, p=1.0), image, seg)
                self.assertTrue(bool(torch.allclose(out.float(), ref, atol=1e-5)))

    def test_rejects_bad_arguments(self):
        bad = (
            {"extend": 1.0},
            {"extend": (0.5, 0.5)},
            {"extend": -0.1},
            {"fill": "perlin"},
            {"fill": []},
            {"fill": ["em", "zeros"]},
            {"regions": (0, 3)},
            {"cells": (5, 3)},
            {"margin": -1},
            {"layers": 0},
            {"border_p": 1.5},
            {"p_batch": 0.5},
        )
        for kwargs in bad:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                RandomFOVWindowGPU(**kwargs)


class TestBlackBorder(SmaugLabTestCase):
    def detect(self, image):
        return _black_border(image, (image == 0.0).all(dim=1, keepdim=True))

    def test_finds_a_constant_level(self):
        image, _ = case()
        border = self.detect(image)
        self.assertTrue(bool(torch.equal(border, (image == BLACK).all(dim=1, keepdim=True))))

    def test_survives_interpolation_jitter_and_a_bin_edge(self):
        image, _ = case()
        for level in (BLACK, 0.0035 - 1e-9, -0.0005):  # the last two sit on a bin edge
            with self.subTest(level=level):
                jittered = image.clone()
                region = jittered[:, :, 6:10]
                region.copy_(level + 1e-7 * torch.randn_like(region))
                expected = torch.zeros_like(jittered, dtype=torch.bool)
                expected[:, :, 6:10] = True
                self.assertTrue(bool(torch.equal(self.detect(jittered), expected)))

    def test_isolated_voxels_at_the_level_are_dropped(self):
        image, _ = case()
        specks = torch.zeros_like(image, dtype=torch.bool)
        specks[:, :, 14:26:3, 2:22:3, 2:18:3] = True
        image[specks] = BLACK
        border = self.detect(image)
        self.assertFalse(bool(border[specks].any()))
        self.assertTrue(bool(border[:, :, 6:10].all()))

    def test_no_border_when_the_level_is_rare_or_bright(self):
        image, _ = case()
        image[:, :, 6:10] = torch.rand_like(image[:, :, 6:10]) + 1.0
        rare = image.clone()
        rare[:, :, 6, :4] = BLACK  # well under 5% of the scan
        self.assertFalse(bool(self.detect(rare).any()))
        bright = image.clone()
        bright[:, :, 6:12] = 5.0  # a constant above the median
        self.assertFalse(bool(self.detect(bright).any()))

    def test_every_channel_must_sit_at_its_level(self):
        image, _ = case((2, 2, 28, 24, 20))
        image[:, 1] = image[:, 0]
        self.assertTrue(bool(self.detect(image)[:, :, 6:10].all()))
        image[:, 1, 6:10] = torch.rand_like(image[:, 1, 6:10]) + 1.0
        self.assertFalse(bool(self.detect(image).any()))

    def test_ct_air_at_the_minimum_counts(self):
        """No CT guard: clipped air is selected like an MRI border."""
        image, _ = case()
        image[:, :, 6:10] = -1.2
        image[:, :, 10:] = image[:, :, 10:].clamp_min(-1.2)
        self.assertTrue(bool(self.detect(image)[:, :, 6:10].all()))
