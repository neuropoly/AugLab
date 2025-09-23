import os
import json
import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from auglab.transforms.cpu.artifact import ArtifactTransform
from auglab.transforms.cpu.contrast import ConvTransform, HistogramEqualTransform, FunctionTransform
from auglab.transforms.cpu.fromSeg import RedistributeTransform
from auglab.transforms.cpu.spatial import SpatialCustomTransform, ShapeTransform

class AugTransforms(ComposeTransforms):
    def __init__(self, json_path: str):
        # Load transform parameters from JSON
        config_path = os.path.join(json_path)
        with open(config_path, 'r') as f:
            self.transform_params = json.load(f)
        self.transforms = self._build_transforms()
        super().__init__(transforms=self.transforms)

    def _build_transforms(self):
        transform_params = self.transform_params
        transforms = []

        # Scharr filter
        conv_params = transform_params.get('ConvTransform', {})
        conv_prob = conv_params.pop('probability', 0.15)
        conv_params['retain_stats'] = transform_params.get('retain_stats', False)
        transforms.append(RandomTransform(
            ConvTransform(**conv_params), apply_probability=conv_prob
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
                p_invert_image=gamma_inv_params.get('p_invert_image', 1),
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
                p_invert_image=gamma_params.get('p_invert_image', 0),
                synchronize_channels=gamma_params.get('synchronize_channels', False),
                p_per_channel=gamma_params.get('p_per_channel', 1),
                p_retain_stats=gamma_params.get('p_retain_stats', 1)
            ), apply_probability=gamma_prob
        ))

        # Apply functions
        func_list = [
            lambda x: torch.log(1 + x),
            torch.sqrt,
            torch.sin,
            torch.exp,
            lambda x: 1/(1 + torch.exp(-x)),
        ]
        func_prob = transform_params.get('FunctionTransform', {}).get('probability', 0.05)
        for func in func_list:
            transforms.append(RandomTransform(
                FunctionTransform(
                    function=func,
                    retain_stats=transform_params.get('retain_stats', False)
                ), apply_probability=func_prob
            ))

        # Histogram manipulations
        hist_prob = transform_params.get('HistogramEqualTransform', {}).get('probability', 0.1)
        transforms.append(RandomTransform(
            HistogramEqualTransform(
                retain_stats=transform_params.get('retain_stats', False)
            ), apply_probability=hist_prob
        ))

        # Redistribute segmentation values
        redist_params = transform_params.get('RedistributeTransform', {})
        redist_prob = redist_params.pop('probability', 0.5)
        redist_params['retain_stats'] = transform_params.get('retain_stats', False)
        transforms.append(RandomTransform(
            RedistributeTransform(
                **redist_params
            ), apply_probability=redist_prob
        ))

        # Resolution transforms
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
        if transform_params.get('mirror_axes') is not None and len(transform_params['mirror_axes']) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=transform_params['mirror_axes']
                )
            )

        # Artifacts generation
        artifact_params = transform_params.get('ArtifactTransform', {})
        artifact_prob = artifact_params.pop('probability', 0.7)
        transforms.append(RandomTransform(
            ArtifactTransform(
                **artifact_params
            ), apply_probability=artifact_prob
        ))

        # Spatial transforms
        spatial_params = transform_params.get('SpatialCustomTransform', {})
        spatial_prob = spatial_params.pop('probability', 0.6)
        transforms.append(RandomTransform(
            SpatialCustomTransform(
                **spatial_params
            ), apply_probability=spatial_prob
        ))

        return transforms

class AugTransformsTest(ComposeTransforms):
    def __init__(self):
        self.transforms = self._build_transforms()
        super().__init__(transforms=self.transforms)

    def _build_transforms(self):
        transforms = []

        # Scharr filter
        transforms.append(RandomTransform(
            ConvTransform(
                kernel_type="Scharr",
                absolute=True,
            ), apply_probability=0.9
        ))

        # Affine transforms
        transforms.append(RandomTransform(
            SpatialCustomTransform(
                affine=True,
            ), apply_probability=0.9
        ))

        return transforms