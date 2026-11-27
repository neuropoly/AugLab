"""`RandomFOVWindowGPU`, the dataloader half and the trainer wiring.

The stub case encodes each voxel's position in its intensity (`1 + x*10^4 + y*10^2 + z`),
so a crop can be checked voxel by voxel against the bbox it was cut from, and nnU-Net's 0
padding is the only 0. The label is a rod along the first axis through both of its faces,
as the spine runs through a sagittal slab's cranial and caudal faces.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import unittest

import numpy as np
import torch

from smauglab.transforms.gpu.context import RandomFOVWindowGPU, WindowSettings

# Raised at import, as in test_trainers.py: the loader is nnU-Net's.
if importlib.util.find_spec("nnunetv2") is None:
    raise unittest.SkipTest("the loader needs the nnunetv2 extra")

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd  # noqa: E402
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402
from nnunetv2.utilities.label_handling.label_handling import LabelManager  # noqa: E402

from smauglab.trainers import fov_window  # noqa: E402

SHAPE = (40, 30, 24)  # the case, (x, y, z)
INITIAL = (32, 32, 32)  # nnU-Net's crop, before SpatialTransform
FINAL = (24, 24, 24)  # the patch the network sees
SEED = 1234

#: sha256 of the nnunetv2 2.6.4 sources this code copies or relies on. A mismatch means
#: nnU-Net changed under it: re-check `fov_window.get_dataloaders` against the new
#: `get_dataloaders`, and that `generate_train_batch` still calls `get_bbox` once per sample.
UPSTREAM_SHA256 = {
    "nnUNetTrainer.get_dataloaders": "c69120a72088ea83026a0bcdd7183ce6441c5a1645e8a3a41846971dcdefd48f",
    "nnUNetDataLoader.get_bbox": "25263f7173ae246b17321c09e388562c8b4b6daaf41754283d4b6091b4302253",
    "nnUNetDataLoader.generate_train_batch": "d1a41fd1541db70eeb09c69bc85d0e70573cd9236ec77aa0f6a5cc2f3a48e7d4",
}


def make_case(shape=SHAPE, rod=True):
    x, y, z = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    data = (1 + x * 10_000 + y * 100 + z).astype(np.float32)[None]
    seg = np.zeros((1, *shape), dtype=np.int16)
    if rod:
        seg[0, :, 12:18, 9:15] = 1  # through both faces of axis 0, near none of the others
    locations = {np.int64(1): np.argwhere(seg > 0)}  # numpy keys, as nnU-Net's preprocessing writes them
    return data, seg, locations


class StubDataset:
    def __init__(self, n_cases=3, **case_kwargs):
        self.identifiers = [f"case_{i}" for i in range(n_cases)]
        self.case = make_case(**case_kwargs)

    def load_case(self, identifier):
        data, seg, locations = self.case
        return data, seg, None, {"class_locations": locations}


def loader(cls, window=None, oversample=0.33, dataset=None, batch_size=2):
    kwargs = {"window": window} if window is not None else {}
    return cls(
        dataset or StubDataset(),
        batch_size,
        INITIAL,
        FINAL,
        LabelManager({"background": 0, "rod": 1}, regions_class_order=None),
        oversample_foreground_percent=oversample,
        **kwargs,
    )


def recorded(dl):
    """Wrap `get_bbox` to record what it returns, as `scripts/trainer_batch_dump.py` does."""
    boxes, original = [], dl.get_bbox

    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        boxes.append(out)
        return out

    dl.get_bbox = wrapper
    return boxes


class TestLoader(unittest.TestCase):
    def test_p_zero_is_nnunet_bit_for_bit(self):
        batches, states = [], []
        for cls, window in ((nnUNetDataLoader, None), (fov_window.FOVWindowDataLoader, WindowSettings(0.0, (0.75,) * 3, 4))):
            np.random.seed(SEED)
            dl = loader(cls, window)
            batches.append([dl.generate_train_batch() for _ in range(20)])
            states.append(np.random.get_state()[1].copy())
        for ref, out in zip(*batches):
            np.testing.assert_array_equal(ref["data"], out["data"])
            np.testing.assert_array_equal(ref["target"], out["target"])
            self.assertEqual(list(ref["keys"]), list(out["keys"]))
            self.assertFalse(bool(out["fov"].any()))
        np.testing.assert_array_equal(states[0], states[1])

    def test_selection_rate_and_flag(self):
        np.random.seed(SEED)
        dl = loader(fov_window.FOVWindowDataLoader, WindowSettings(0.5, (0.75,) * 3, 4))
        flags = torch.cat([dl.generate_train_batch()["fov"] for _ in range(200)])
        self.assertEqual(flags.dtype, torch.bool)
        self.assertGreater(float(flags.float().mean()), 0.4)
        self.assertLess(float(flags.float().mean()), 0.6)

    def test_crops_are_the_case_at_the_moved_bbox(self):
        np.random.seed(SEED)
        dl = loader(fov_window.FOVWindowDataLoader, WindowSettings(1.0, (0.75,) * 3, 4))
        boxes = recorded(dl)
        data, seg, _ = make_case()
        for _ in range(10):
            batch = dl.generate_train_batch()
            for b in range(2):
                lbs, ubs = boxes.pop(0)
                bbox = [[lo, hi] for lo, hi in zip(lbs, ubs)]
                np.testing.assert_array_equal(batch["data"][b], crop_and_pad_nd(data, bbox, 0))
                np.testing.assert_array_equal(batch["target"][b], crop_and_pad_nd(seg, bbox, -1))

    def test_windows_leave_the_case_only_through_faces_no_label_touches(self):
        np.random.seed(SEED)
        dl = loader(fov_window.FOVWindowDataLoader, WindowSettings(1.0, (0.75,) * 3, 4), oversample=0.0)
        boxes = recorded(dl)
        for _ in range(100):
            dl.generate_train_batch()
        lbs = np.array([b[0] for b in boxes])
        start = lbs + (np.array(INITIAL) - np.array(FINAL)) // 2  # SpatialTransform's centre crop
        end = start + np.array(FINAL)
        # Axis 0: the rod runs through both faces, so the patch stays inside, as nnU-Net keeps it.
        self.assertGreaterEqual(int(start[:, 0].min()), 0)
        self.assertLessEqual(int(end[:, 0].max()), SHAPE[0])
        # Axes 1 and 2: the case is extended, so some windows leave it on each side.
        for axis in (1, 2):
            self.assertLess(int(start[:, axis].min()), 0, axis)
            self.assertGreater(int(end[:, axis].max()), SHAPE[axis], axis)

    def test_a_random_window_keeps_one_minus_extend_of_the_patch_in_the_case(self):
        np.random.seed(SEED)
        extend = (0.75, 0.5, 0.25)
        dl = loader(fov_window.FOVWindowDataLoader, WindowSettings(1.0, extend, 4), oversample=0.0, dataset=StubDataset(rod=False))
        boxes = recorded(dl)
        for _ in range(200):
            dl.generate_train_batch()
        start = np.array([b[0] for b in boxes]) + (np.array(INITIAL) - np.array(FINAL)) // 2
        overlap = np.minimum(start + np.array(FINAL), SHAPE) - np.maximum(start, 0)
        floor = np.array(FINAL) - np.round(np.array(extend) * np.array(FINAL))
        self.assertTrue(bool((overlap >= floor).all()), (overlap.min(0), floor))
        self.assertTrue(bool((overlap.min(0) <= floor + 2).all()), (overlap.min(0), floor))  # and the extension is used

    def test_the_oversampled_row_stays_on_a_label(self):
        np.random.seed(SEED)
        dl = loader(fov_window.FOVWindowDataLoader, WindowSettings(1.0, (0.75,) * 3, 4), oversample=1.0)
        for _ in range(20):
            batch = dl.generate_train_batch()
            self.assertTrue(bool((batch["target"] > 0).reshape(2, -1).any(1).all()))

    def test_2d_is_refused(self):
        with self.assertRaises(ValueError):
            fov_window.FOVWindowDataLoader(
                StubDataset(),
                2,
                (32, 32),
                (24, 24),
                LabelManager({"background": 0, "rod": 1}, regions_class_order=None),
                window=WindowSettings(0.5, (0.5,) * 3, 4),
            )


class TestTouchedFaces(unittest.TestCase):
    def test_tolerance_and_faces(self):
        locations = {1: np.array([[0, 3, 10, 0], [0, 30, 12, 23]])}
        touched = fov_window.touched_case_faces(locations, (40, 30, 24), tol=3)
        np.testing.assert_array_equal(touched, [[True, False], [False, False], [True, True]])
        self.assertFalse(fov_window.touched_case_faces(locations, (40, 30, 24), tol=2)[0, 0])

    def test_numpy_keys_beside_the_annotated_tuple(self):
        """nnU-Net writes numpy-integer keys; `np.int64(1) != (-1, 1)` compares elementwise."""
        locations = {np.int64(1): np.array([[0, 4, 4, 4]]), (-1, 1): np.array([[0, 0, 0, 0]])}
        self.assertFalse(fov_window.touched_case_faces(locations, (9, 9, 9), tol=1, skip_key=(-1, 1)).any())
        self.assertTrue(fov_window.touched_case_faces(locations, (9, 9, 9), tol=1)[:, 0].all())  # the tuple entry counts without skip_key

    def test_no_labels_and_the_annotated_key(self):
        self.assertFalse(fov_window.touched_case_faces({1: np.zeros((0, 4))}, (8, 8, 8), tol=4).any())
        self.assertFalse(fov_window.touched_case_faces(None, (8, 8, 8), tol=4).any())
        locations = {(-1, 0, 1): np.array([[0, 0, 0, 0]]), 1: np.array([[0, 4, 4, 4]])}
        self.assertFalse(fov_window.touched_case_faces(locations, (9, 9, 9), tol=1, skip_key=(-1, 0, 1)).any())


class TestUpstreamPins(unittest.TestCase):
    def test_the_upstream_sources_are_the_ones_copied(self):
        for name, method in (
            ("nnUNetTrainer.get_dataloaders", nnUNetTrainer.get_dataloaders),
            ("nnUNetDataLoader.get_bbox", nnUNetDataLoader.get_bbox),
            ("nnUNetDataLoader.generate_train_batch", nnUNetDataLoader.generate_train_batch),
        ):
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(inspect.getsource(method).encode()).hexdigest(), UPSTREAM_SHA256[name])

    def test_the_copy_differs_from_upstream_only_in_the_training_loader(self):
        def normalised(lines):
            return [" ".join(ln.split()) for ln in lines if ln.strip()]

        upstream = normalised(inspect.getsource(nnUNetTrainer.get_dataloaders).splitlines()[1:])  # without `def`
        ours = inspect.getsource(fov_window.get_dataloaders).splitlines()
        docstring_end = next(i for i in range(2, len(ours)) if ours[i].strip().endswith('"""'))
        ours = normalised(ours[docstring_end + 1 :])
        # Undo the one intended change: the training loader's class and its `window` argument.
        i = ours.index("window=window)")
        ours[i - 1] = ours[i - 1].removesuffix(",") + ")"
        del ours[i]
        ours = [ln.replace("dl_tr = FOVWindowDataLoader(", "dl_tr = nnUNetDataLoader(") for ln in ours]
        self.assertEqual(ours, upstream)


class _Pipeline(torch.nn.Module):
    """Stands in for the GPU pipeline: holds the transform and records the selection it was handed."""

    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        self.seen = []

    def forward(self, data, target):
        self.seen.append(self.transform._p_gen.pending)
        self.transform._p_gen.pending = None
        return data, target


class TestTrainerWiring(unittest.TestCase):
    def _trainer(self, pipeline):
        from smauglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

        trainer = nnUNetTrainerDAExtGPU.__new__(nnUNetTrainerDAExtGPU)
        trainer.device = torch.device("cpu")
        trainer.transforms = pipeline
        trainer._fov_window = fov_window.find_fov_window(pipeline)
        trainer.grad_scaler = None
        trainer.network = torch.nn.Conv3d(1, 2, 1)
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.1)
        trainer.loss = lambda output, _target: output.mean()
        trainer._get_deep_supervision_scales = lambda: None
        return trainer

    def _batch(self, fov=None):
        batch = {"data": torch.rand(2, 1, 4, 4, 4), "target": torch.zeros(2, 1, 4, 4, 4), "keys": ["a", "b"]}
        if fov is not None:
            batch["fov"] = torch.tensor(fov)
        return batch

    def test_train_step_hands_the_loaders_rows_to_the_transform(self):
        pipeline = _Pipeline(RandomFOVWindowGPU(p=0.5))
        trainer = self._trainer(pipeline)
        trainer.train_step(self._batch([True, False]))
        self.assertEqual(pipeline.seen[0].tolist(), [True, False])

    def test_a_batch_without_the_rows_is_an_error(self):
        trainer = self._trainer(_Pipeline(RandomFOVWindowGPU(p=0.5)))
        with self.assertRaisesRegex(KeyError, "FOVWindowDataLoader"):
            trainer.train_step(self._batch())

    def test_without_the_transform_nothing_changes(self):
        self.assertIsNone(fov_window.find_fov_window(None))
        self.assertIsNone(fov_window.find_fov_window(torch.nn.Sequential(torch.nn.Identity())))
        trainer = self._trainer(None)
        self.assertIsNone(trainer._fov_window)
        trainer.train_step(self._batch())

    def test_two_windows_are_refused(self):
        with self.assertRaises(ValueError):
            fov_window.find_fov_window(torch.nn.Sequential(RandomFOVWindowGPU(), RandomFOVWindowGPU()))

    def test_the_class_default_is_none(self):
        from smauglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

        self.assertIsNone(nnUNetTrainerDAExtGPU.__new__(nnUNetTrainerDAExtGPU)._fov_window)
