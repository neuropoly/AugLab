from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from typing import Tuple, Union, List
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

import os
import torch
import importlib
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from auglab.transforms.cpu.transforms import AugTransforms
import auglab.configs as configs
from auglab.transforms.gpu.transforms import AugTransformsGPU
from auglab.trainers.utils import DownsampleSegForDSTransformCustom

class nnUNetTrainerDAExt(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
    
    def get_dataloaders(self):
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

        # # Deactivate default mirroring data augmentation from nnUNetTrainer
        # mirror_axes = None
        # self.inference_allowed_mirroring_axes = None

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

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
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
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        ### Adds transforms
        configs_path = importlib.resources.files(configs)
        json_path = configs_path / "transform_params.json"
        transforms.append(AugTransforms(json_path=str(json_path)))

        ## Removed do_dummy_2d_data_aug
        
        ## Unclear what this does
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
    
    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

class nnUNetTrainerDAExtGPU(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_GPU_JSON", str(configs_path / "transform_params_gpu.json"))
        self.transforms = AugTransformsGPU(json_path=json_path).to(self.device)
        print(f'Using AugLab GPU transforms with parameters from: {json_path}')

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        ### Keep some nnunet transforms
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, 
                patch_center_dist_from_border=0, 
                random_crop=False, 
                p_elastic_deform=0,
                p_rotation=0,
                rotation=rotation_for_DA, 
                p_scaling=0, 
                scaling=(0.7, 1.4), 
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False, 
                mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        # Now target should be a single tensor, not a list
        target = target.to(self.device, non_blocking=True)
        # if isinstance(target, list):
        #     target = [i.to(self.device, non_blocking=True) for i in target]
        # else:
        #     target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():            
            # Apply GPU augmentations to full-resolution data/target
            data, target = self.transforms(data, target)
    
            # Create multi-scale targets for deep supervision after augmentation
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                ds_transform = DownsampleSegForDSTransformCustom(ds_scales=deep_supervision_scales)
                target = ds_transform(target)
    
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

