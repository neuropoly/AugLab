from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from typing import Tuple, Union, List
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

import torch

from auglab.transforms.artifact import ArtifactTransform
from auglab.transforms.contrast import ConvTransform, HistogramEqualTransform, FunctionTransform
from auglab.transforms.fromSeg import RedistributeTransform
from auglab.transforms.spatial import SpatialCustomTransform, ShapeTransform
import json
import os

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

        # Load transform parameters from JSON
        config_path = os.path.join(os.path.dirname(__file__), '../configs/transform_params.json')
        with open(config_path, 'r') as f:
            transform_params = json.load(f)

        ### Adds custom nnunet transforms
        ## Image transforms
        # Scharr filter
        conv_params = transform_params.get('ConvTransform', {})
        conv_prob = conv_params.pop('probability', 0.15)
        conv_params['retain_stats'] = retain_stats  # allow override if needed
        transforms.append(RandomTransform(
            ConvTransform(
                **conv_params
            ), apply_probability=conv_prob
        ))
        
        # Gaussian blur
        blur_params = transform_params.get('GaussianBlurTransform', {})
        blur_prob = blur_params.pop('probability', 0.2)
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=tuple(blur_params.get('blur_sigma', (0.5, 1.))),
                synchronize_channels=blur_params.get('synchronize_channels', False),
                synchronize_axes=blur_params.get('synchronize_axes', False),
                p_per_channel=blur_params.get('p_per_channel', 0.5),
                benchmark=blur_params.get('benchmark', True)
            ), apply_probability=blur_prob
        ))

        # Noise transforms
        noise_params = transform_params.get('GaussianNoiseTransform', {})
        noise_prob = noise_params.pop('probability', 0.1)
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=tuple(noise_params.get('noise_variance', (0, 0.1))),
                p_per_channel=noise_params.get('p_per_channel', 1),
                synchronize_channels=noise_params.get('synchronize_channels', True)
            ), apply_probability=noise_prob
        ))

        # Brightness transforms
        bright_params = transform_params.get('MultiplicativeBrightnessTransform', {})
        bright_prob = bright_params.pop('probability', 0.15)
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast(tuple(bright_params.get('multiplier_range', (0.75, 1.25)))),
                synchronize_channels=bright_params.get('synchronize_channels', False),
                p_per_channel=bright_params.get('p_per_channel', 1)
            ), apply_probability=bright_prob
        ))

        # Contrast transforms
        contrast_params = transform_params.get('ContrastTransform', {})
        contrast_prob = contrast_params.pop('probability', 0.15)
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast(tuple(contrast_params.get('contrast_range', (0.75, 1.25)))),
                preserve_range=contrast_params.get('preserve_range', True),
                synchronize_channels=contrast_params.get('synchronize_channels', False),
                p_per_channel=contrast_params.get('p_per_channel', 1)
            ), apply_probability=contrast_prob
        ))

        # Gamma transforms
        gamma_inv_params = transform_params.get('GammaTransform_invert', {})
        gamma_inv_prob = gamma_inv_params.pop('probability', 0.1)
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast(tuple(gamma_inv_params.get('gamma', (0.7, 1.5)))),
                p_invert_image=gamma_inv_params.get('p_invert_image', 1), # With inversion
                synchronize_channels=gamma_inv_params.get('synchronize_channels', False),
                p_per_channel=gamma_inv_params.get('p_per_channel', 1),
                p_retain_stats=gamma_inv_params.get('p_retain_stats', 1)
            ), apply_probability=gamma_inv_prob
        ))

        gamma_params = transform_params.get('GammaTransform', {})
        gamma_prob = gamma_params.pop('probability', 0.3)
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast(tuple(gamma_params.get('gamma', (0.7, 1.5)))),
                p_invert_image=gamma_params.get('p_invert_image', 0), # Without inversion
                synchronize_channels=gamma_params.get('synchronize_channels', False),
                p_per_channel=gamma_params.get('p_per_channel', 1),
                p_retain_stats=gamma_params.get('p_retain_stats', 1)
            ), apply_probability=gamma_prob
        ))

        # Apply functions
        func_list = [
            lambda x: torch.log(1 + x), # Log
            torch.sqrt, # sqrt
            torch.sin, # sin
            torch.exp, # exp
            lambda x: 1/(1 + torch.exp(-x)), # sig
        ]

        func_prob = transform_params.get('FunctionTransform', {}).get('probability', 0.05)
        for func in func_list:
            transforms.append(RandomTransform(
                FunctionTransform(
                    function=func,
                    retain_stats=retain_stats
                ), apply_probability=func_prob
            ))

        # Histogram manipulations
        hist_prob = transform_params.get('HistogramEqualTransform', {}).get('probability', 0.1)
        transforms.append(RandomTransform(
            HistogramEqualTransform(
                retain_stats=retain_stats
            ), apply_probability=hist_prob
        ))

        # Redistribute segmentation values
        redist_params = transform_params.get('RedistributeTransform', {})
        redist_prob = redist_params.pop('probability', 0.5)
        redist_params['retain_stats'] = retain_stats
        transforms.append(RandomTransform(
            RedistributeTransform(
                **redist_params
            ), apply_probability=redist_prob
        ))

        ## Resolution transforms
        # Simulate reduced shape
        shape_params = transform_params.get('ShapeTransform', {})
        shape_prob = shape_params.pop('probability', 0.4)
        transforms.append(RandomTransform(
            ShapeTransform(
                **shape_params
            ), apply_probability=shape_prob
        ))

        # Simulate low resolution
        lowres_params = transform_params.get('SimulateLowResolutionTransform', {})
        lowres_prob = lowres_params.pop('probability', 0.2)
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=tuple(lowres_params.get('scale', (0.3, 1))),
                synchronize_channels=lowres_params.get('synchronize_channels', True),
                synchronize_axes=lowres_params.get('synchronize_axes', False),
                ignore_axes=tuple(lowres_params.get('ignore_axes', ())),
                allowed_channels=lowres_params.get('allowed_channels', None),
                p_per_channel=lowres_params.get('p_per_channel', 0.5)
            ), apply_probability=lowres_prob
        ))

        # Mirroring transforms
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        ## Artifacts generation
        artifact_params = transform_params.get('ArtifactTransform', {})
        artifact_prob = artifact_params.pop('probability', 0.7)
        transforms.append(RandomTransform(
            ArtifactTransform(
                **artifact_params
            ), apply_probability=artifact_prob
        ))

        ## Spatial transforms
        spatial_params = transform_params.get('SpatialCustomTransform', {})
        spatial_prob = spatial_params.pop('probability', 0.6)
        transforms.append(RandomTransform(
            SpatialCustomTransform(
                **spatial_params
            ), apply_probability=spatial_prob
        ))
        '''
        # Use nnunet transform instead?
        SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False#, mode_seg='nearest'
            )
        '''
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
