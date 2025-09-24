import os, json

from kornia.augmentation import AugmentationSequential
import torch.nn as nn

from auglab.transforms.gpu.contrast import RandomConvTransformGPU
from auglab.transforms.gpu.spatial import RandomAffine3DCustom
from auglab.transforms.gpu.base import AugmentationSequentialCustom

class AugTransformsGPU(AugmentationSequentialCustom):
    """
    Module to perform data augmentation on GPU.
    """
    def __init__(self, json_path: str):
        # Load transform parameters from JSON
        config_path = os.path.join(json_path)
        with open(config_path, 'r') as f:
            self.transform_params = json.load(f)
        transforms = self._build_transforms()
        super().__init__(*transforms, data_keys=["input", "mask"], same_on_batch=False)

    def _build_transforms(self) -> list[nn.Module]:
        transforms = []

        # Scharr filter
        # conv_params = self.transform_params.get('ConvTransform')
        # transforms.append(RandomConvTransformGPU(
        #     kernel_type=conv_params['kernel_type'],
        #     absolute=conv_params['absolute'],
        #     p=conv_params['probability']
        # ))

        # Gaussian blur
        

        # Noise transforms

        # Brightness transforms

        # Contrast transforms

        # Gamma transforms

        # Apply functions

        # Histogram manipulations

        # Redistribute segmentation values

        # Resolution transforms

        # Simulate low resolution

        # Mirroring transforms

        # Artifacts generation

        # Spatial transforms
        affine_params = self.transform_params.get('AffineTransform')
        transforms.append(RandomAffine3DCustom(
            degrees=affine_params.get('degrees'),
            translate=affine_params.get('translate'),
            scale=affine_params.get('scale'),
            shears=None, #affine_params.get('shear'),
            resample='bilinear',
            p=affine_params.get('probability'),
            keepdim=False
        ))
        return transforms


