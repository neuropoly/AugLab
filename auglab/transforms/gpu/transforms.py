import os, json

import torch.nn as nn
import torch

from auglab.transforms.gpu.contrast import RandomConvTransformGPU, RandomGaussianNoiseGPU, RandomBrightnessGPU, RandomGammaGPU, RandomFunctionGPU, \
RandomHistogramEqualizationGPU, RandomInverseGPU
from auglab.transforms.gpu.spatial import RandomAffine3DCustom, RandomLowResTransformGPU, RandomFlipTransformGPU
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
        scharr_params = self.transform_params.get('ScharrTransform')
        if scharr_params is not None:
            transforms.append(RandomConvTransformGPU(
                kernel_type=scharr_params.get('kernel_type', 'Scharr'),
                p=scharr_params.get('probability', 0),
                retain_stats=scharr_params.get('retain_stats', True),
                absolute=scharr_params.get('absolute', True),
            ))

        # Gaussian blur
        gaussianblur_params = self.transform_params.get('GaussianBlurTransform')
        if gaussianblur_params is not None:
            transforms.append(RandomConvTransformGPU(
                kernel_type=gaussianblur_params.get('kernel_type', 'GaussianBlur'),
                p=gaussianblur_params.get('probability', 0),
                sigma=gaussianblur_params.get('sigma', 1.0),
            ))

        # Unsharp masking
        unsharp_params = self.transform_params.get('UnsharpMaskTransform')
        if unsharp_params is not None:
            transforms.append(RandomConvTransformGPU(
                kernel_type=unsharp_params.get('kernel_type', 'UnsharpMask'),
                p=unsharp_params.get('probability', 0),
                sigma=unsharp_params.get('sigma', 1.0),
                unsharp_amount=unsharp_params.get('unsharp_amount', 1.5),
        ))

        # Noise transforms
        noise_params = self.transform_params.get('GaussianNoiseTransform')
        if noise_params is not None:
            transforms.append(RandomGaussianNoiseGPU(
                mean=noise_params.get('mean', 0.0),
                std=noise_params.get('std', 1.0),
                p=noise_params.get('probability', 0),
            ))

        # Brightness transforms
        brightness_params = self.transform_params.get('BrightnessTransform')
        if brightness_params is not None:
            transforms.append(RandomBrightnessGPU(
                brightness_range=brightness_params.get('brightness_range', [0.5, 1.5]),
                p=brightness_params.get('probability', 0),
            ))

        # Gamma transforms
        gamma_params = self.transform_params.get('GammaTransform')
        if gamma_params is not None:
            transforms.append(RandomGammaGPU(
                gamma_range=gamma_params.get('gamma_range', [0.7, 1.5]),
                p=gamma_params.get('probability', 0),
                invert_image=False,
                retain_stats=gamma_params.get('retain_stats', False),
            ))

            transforms.append(RandomGammaGPU(
                gamma_range=gamma_params.get('gamma_range', [0.7, 1.5]),
                p=gamma_params.get('probability', 0),
                invert_image=True,
                retain_stats=gamma_params.get('retain_stats', False),
            ))

        # Apply functions
        func_list = [
            lambda x: torch.log(1 + x),
            torch.sqrt,
            torch.sin,
            torch.exp,
            lambda x: 1/(1 + torch.exp(-x)),
        ]
        function_params = self.transform_params.get('FunctionTransform')
        if function_params is not None:
            for func in func_list:
                transforms.append(RandomFunctionGPU(
                    func=func,
                    p=function_params.get('probability', 0),
                    retain_stats=function_params.get('retain_stats', False),
            ))
        
        # Inverse transform (max - pixel_value)
        inverse_params = self.transform_params.get('InverseTransform')
        if inverse_params is not None:
            transforms.append(RandomInverseGPU(
                p=inverse_params.get('probability', 0),
                retain_stats=inverse_params.get('retain_stats', False),
            ))
        
        # Histogram manipulations
        histo_params = self.transform_params.get('HistogramEqualizationTransform')
        if histo_params is not None:
            transforms.append(RandomHistogramEqualizationGPU(
                p=histo_params.get('probability', 0),
                retain_stats=histo_params.get('retain_stats', False),
            ))
        
        # Redistribute segmentation values (Not implemented on GPU yet)

        # Shape transforms (Cropping and Simulating low resolution)
        shape_params = self.transform_params.get('ShapeTransform')
        if shape_params is not None:
            transforms.append(RandomLowResTransformGPU(
                p=shape_params.get('probability', 0),
                scale=shape_params.get('scale', [0.3, 1.0]),
                crop=shape_params.get('crop', [1.0, 1.0]),
                same_on_batch=shape_params.get('same_on_batch', False)
        ))

        # Flipping transforms
        flip_params = self.transform_params.get('FlipTransform')
        if flip_params is not None:
            transforms.append(RandomFlipTransformGPU(
                flip_axis=flip_params.get('flip_axis', [0]),
                p=flip_params.get('probability', 0),
                same_on_batch=flip_params.get('same_on_batch', False),
                keepdim=flip_params.get('keepdim', True)
            ))

        # Artifacts generation (Not implemented on GPU yet)

        # Spatial transforms
        affine_params = self.transform_params.get('AffineTransform')
        if affine_params is not None:
            transforms.append(RandomAffine3DCustom(
                degrees=affine_params.get('degrees', 10),
                translate=affine_params.get('translate', [0.1, 0.1, 0.1]),
                scale=affine_params.get('scale', [0.9, 1.1]),
                shears=affine_params.get('shear', [-10, 10, -10, 10, -10, 10]),
                resample=affine_params.get('resample', "bilinear"),
                p=affine_params.get('probability', 0)
            ))

        # Elastic transforms (Not implemented on GPU yet)

        return transforms

if __name__ == "__main__":
    # Example usage
    import importlib
    import auglab.configs as configs
    from auglab.utils.image import Image

    configs_path = importlib.resources.files(configs)
    json_path = configs_path / "transform_params_gpu.json"
    augmentor = AugTransformsGPU(json_path)

    # Load image and mask tensors
    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/sub-amu02/anat/sub-amu02_T1w.nii.gz'
    img = Image(img_path).change_orientation('RSP')
    img_tensor = torch.from_numpy(img.data.copy()).unsqueeze(0).to(torch.float32)

    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/derivatives/labels/sub-amu02/anat/sub-amu02_T1w_label-spine_dseg.nii.gz'
    seg = Image(seg_path).change_orientation('RSP')
    seg_tensor_all = torch.from_numpy(seg.data.copy()).unsqueeze(0)

    # Add segmentation values to different channels
    seg_tensor = torch.zeros((5, *seg_tensor_all.shape[1:]))
    for i, value in enumerate([12, 13, 14, 15, 16]):
        seg_tensor[i] = (seg_tensor_all == value)

    # Format tensors to match expected input shape (B, C, D, H, W)
    img_tensor = torch.cat([img_tensor, seg_tensor_all.bool().int()], dim=0).unsqueeze(0)  # Add batch dimension and seconda channel
    seg_tensor = seg_tensor.unsqueeze(0)  # Add batch dimension

    # Add same image in batch 
    img_tensor = torch.cat([img_tensor, img_tensor], dim=0)
    seg_tensor = torch.cat([seg_tensor, seg_tensor], dim=0)

    # Move to GPU
    img_tensor = img_tensor.cuda()
    seg_tensor = seg_tensor.cuda()
    augmentor = augmentor.cuda()

    # Apply augmentations
    augmented_img, augmented_seg = augmentor(img_tensor.clone(), seg_tensor.clone())

    if augmented_img.shape != img_tensor.shape:
        raise ValueError("Augmented image shape does not match input shape.")
    if augmented_seg.shape != seg_tensor.shape:
        raise ValueError("Augmented segmentation shape does not match input shape.")
    
    import cv2
    import numpy as np
    import warnings, sys, os
    warnings.simplefilter("always")

    # Convert tensors to numpy arrays
    img_tensor_np = img_tensor.cpu().detach().numpy()
    seg_tensor_np = seg_tensor.cpu().detach().numpy()
    augmented_img_np = augmented_img.cpu().detach().numpy()
    augmented_seg_np = augmented_seg.cpu().detach().numpy()

    # Concatenate segmentation channels for visualization
    seg_tensor_np = np.sum(seg_tensor_np, axis=1)
    augmented_seg_np = np.sum(augmented_seg_np, axis=1)

    # Save the augmented images
    middle_slice = img_tensor_np.shape[2] // 2
    os.makedirs('img', exist_ok=True)
    cv2.imwrite('img/augmented_img.png', augmented_img_np[0, 0, middle_slice])
    cv2.imwrite('img/augmented_img2.png', augmented_img_np[1, 0, middle_slice])
    cv2.imwrite('img/not_augmented_channel.png', augmented_img_np[0, 1, middle_slice]*255)
    cv2.imwrite('img/img.png', img_tensor_np[0, 0, middle_slice])
    cv2.imwrite('img/augmented_seg.png', augmented_seg_np[0, middle_slice]*255)
    cv2.imwrite('img/seg.png', seg_tensor_np[0, middle_slice]*255)

    print(augmentor)