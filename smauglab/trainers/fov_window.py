"""The dataloader half of `RandomFOVWindowGPU`: place the training window in a case nnU-Net is told is larger.

nnU-Net 2.6.4 draws each training crop in `nnUNetDataLoader.get_bbox` from bounds that keep
the final patch inside the case (a random window) or centre it on a label voxel (the
oversampled one), then crops with `crop_and_pad_nd`, padding the image with 0 and the
segmentation with -1. Its `pad_sides` already lets it pretend the case is larger, but for
every sample and on every face. `FOVWindowDataLoader` does the same per sample, with
probability `p`, and only on the faces no label comes near: it hands `get_bbox` a case
padded by `extend` of the final patch on those faces (and the class locations shifted to
match) and maps the bbox back. nnU-Net's own sampler, foreground oversampling included,
then places the window, the patch holds real data wherever the case has it, and the GPU
half fills only what lies outside the case. Which samples were moved goes to `train_step`
as `batch["fov"]`, so the fill lands on exactly those.

A face a label touches is never extended: the spine runs through the case's cranial and
caudal faces, and synthetic tissue beyond them would give it an end it does not have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from smauglab.transforms.gpu.context import RandomFOVWindowGPU, WindowSettings

if TYPE_CHECKING:
    from torch.nn import Module

#: Voxels added to `margin` when deciding from `class_locations` whether a label touches a
#: case face. Those are a sample of at most 10k voxels per class, so their extent can fall
#: short of the label's. At the faces labels touch, it fell short by at most 2 voxels on
#: Datasets 803/806/807 and 4 on 804, and no touched face was missed (every case of the
#: four, `transform_samples/fov_window_803-807/touched.json`). A missed face still gets no
#: fill within `margin` of the label.
FACE_SLACK = 4


def find_fov_window(pipeline: Module | None) -> RandomFOVWindowGPU | None:
    """The pipeline's `RandomFOVWindowGPU`, or None. More than one would split the loader's selection."""
    if pipeline is None:
        return None
    found = [m for m in pipeline.modules() if isinstance(m, RandomFOVWindowGPU)]
    if len(found) > 1:
        raise ValueError(f"the GPU pipeline holds {len(found)} RandomFOVWindowGPU; the dataloader can serve one")
    return found[0] if found else None


def touched_case_faces(class_locations: dict | None, data_shape, tol: int, skip_key=None) -> np.ndarray:
    """`[3, 2]` bool: whether any label lies within `tol` voxels of each axis's low and high case face.

    `class_locations` is nnU-Net's per-class sample of label voxels, rows `(channel, x, y, z)`.
    `skip_key` is the ignore-label datasets' "annotated" entry, which is not a label.
    """
    # Keys are numpy integers, or tuples for regions; `np.int64 != tuple` compares elementwise.
    coords = [np.asarray(v)[:, 1:] for k, v in (class_locations or {}).items() if not (isinstance(k, tuple) and k == skip_key) and len(v)]
    if not coords:
        return np.zeros((3, 2), dtype=bool)
    stacked = np.concatenate(coords)
    shape = np.asarray(data_shape)
    return np.stack([stacked.min(0) <= tol, stacked.max(0) >= shape - 1 - tol], axis=1)


class FOVWindowDataLoader(nnUNetDataLoader):
    """`nnUNetDataLoader` that, per sample with probability `window.p`, draws the bbox in a virtually padded case.

    At `p == 0` it draws no extra random number, so its batches are nnU-Net's, bit for bit.
    """

    def __init__(self, *args, window: WindowSettings, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.patch_size_was_2d:
            raise ValueError("RandomFOVWindowGPU's loader is 3-D only")
        self.window = window
        self.extension = np.round(np.asarray(window.extend) * np.asarray(self.final_patch_size)).astype(int)
        self._rows: list[bool] = []

    def get_bbox(self, data_shape, force_fg: bool, class_locations, overwrite_class=None, verbose: bool = False):
        if not (self.window.p > 0 and np.random.uniform() < self.window.p):
            self._rows.append(False)
            return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)
        self._rows.append(True)
        touched = touched_case_faces(class_locations, data_shape, self.window.margin + FACE_SLACK, self.annotated_classes_key)
        low = np.where(touched[:, 0], 0, self.extension)
        high = np.where(touched[:, 1], 0, self.extension)
        virtual = tuple(int(s) for s in np.asarray(data_shape) + low + high)
        offset = np.concatenate([[0], low])
        shifted = None if class_locations is None else {k: (np.asarray(v) + offset if len(v) else v) for k, v in class_locations.items()}
        lbs, ubs = super().get_bbox(virtual, force_fg, shifted, overwrite_class, verbose)
        return [int(b - o) for b, o in zip(lbs, low)], [int(b - o) for b, o in zip(ubs, low)]

    def generate_train_batch(self):
        self._rows = []
        batch = super().generate_train_batch()
        if len(self._rows) != self.batch_size:
            raise RuntimeError(f"nnU-Net drew {len(self._rows)} bboxes for a batch of {self.batch_size}")
        batch["fov"] = torch.tensor(self._rows, dtype=torch.bool)
        return batch


# fmt: off
def get_dataloaders(self, window: WindowSettings):
    """`nnUNetTrainer.get_dataloaders` of nnunetv2 2.6.4, verbatim but for `dl_tr`, which is a `FOVWindowDataLoader`.

    A copy because upstream builds both loaders and starts their workers in one method, so
    neither can be swapped afterwards (`nnUNetTrainerDA5.get_dataloaders` does the same).
    `unit_tests/test_fov_window_loader.py` pins the upstream source and this diff.
    """
    if self.dataset_class is None:
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

    # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
    # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
    patch_size = self.configuration_manager.patch_size

    # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
    # outputs?
    deep_supervision_scales = self._get_deep_supervision_scales()

    (
        rotation_for_DA,
        do_dummy_2d_data_aug,
        initial_patch_size,
        mirror_axes,
    ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

    # training pipeline
    tr_transforms = self.get_training_transforms(
        patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        ignore_label=self.label_manager.ignore_label)

    # validation pipeline
    val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                    is_cascaded=self.is_cascaded,
                                                    foreground_labels=self.label_manager.foreground_labels,
                                                    regions=self.label_manager.foreground_regions if
                                                    self.label_manager.has_regions else None,
                                                    ignore_label=self.label_manager.ignore_label)

    dataset_tr, dataset_val = self.get_tr_and_val_datasets()

    dl_tr = FOVWindowDataLoader(dataset_tr, self.batch_size,
                                initial_patch_size,
                                self.configuration_manager.patch_size,
                                self.label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                probabilistic_oversampling=self.probabilistic_oversampling,
                                window=window)
    dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                              self.configuration_manager.patch_size,
                              self.configuration_manager.patch_size,
                              self.label_manager,
                              oversample_foreground_percent=self.oversample_foreground_percent,
                              sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                              probabilistic_oversampling=self.probabilistic_oversampling)

    allowed_num_processes = get_allowed_n_proc_DA()
    if allowed_num_processes == 0:
        mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
    else:
        mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                    num_processes=allowed_num_processes,
                                                    num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                    pin_memory=self.device.type == 'cuda', wait_time=0.002)
        mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                  transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                  num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                  pin_memory=self.device.type == 'cuda',
                                                  wait_time=0.002)
    # # let's get this party started
    _ = next(mt_gen_train)
    _ = next(mt_gen_val)
    return mt_gen_train, mt_gen_val
# fmt: on
