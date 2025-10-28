import os, json

from kornia.augmentation import AugmentationSequential
import torch.nn as nn

from auglab.transforms.gpu.contrast import RandomConvTransformGPU, RandomGaussianNoiseGPU, RandomBrightnessGPU
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
        scharr_params = self.transform_params.get('ScharrTransform')
        transforms.append(RandomConvTransformGPU(
            kernel_type=scharr_params['kernel_type'],
            p=scharr_params['probability'],
            retain_stats=scharr_params['retain_stats'],
            absolute=scharr_params['absolute'],
        ))

        # Gaussian blur
        gaussianblur_params = self.transform_params.get('GaussianBlurTransform')
        transforms.append(RandomConvTransformGPU(
            kernel_type=gaussianblur_params['kernel_type'],
            p=gaussianblur_params['probability'],
            sigma=gaussianblur_params['sigma'],
        ))

        # Unsharp masking
        unsharp_params = self.transform_params.get('UnsharpMaskTransform')
        transforms.append(RandomConvTransformGPU(
            kernel_type=unsharp_params['kernel_type'],
            p=unsharp_params['probability'],
            sigma=unsharp_params['sigma'],
            unsharp_amount=unsharp_params['unsharp_amount'],
        ))

        # Noise transforms
        noise_params = self.transform_params.get('GaussianNoiseTransform')
        transforms.append(RandomGaussianNoiseGPU(
            mean=noise_params['mean'],
            std=noise_params['std'],
            p=noise_params['probability'],
        ))

        # Brightness transforms
        brightness_params = self.transform_params.get('BrightnessTransform')
        transforms.append(RandomBrightnessGPU(
            brightness_range=brightness_params['brightness_range'],
            p=brightness_params['probability'],
        ))

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
            shears=affine_params.get('shear'),
            resample='bilinear',
            p=affine_params.get('probability'),
            keepdim=False
        ))
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
    import torch
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
    cv2.imwrite('img/not_augmented_channel.png', augmented_img_np[0, 1, middle_slice]*255)
    cv2.imwrite('img/img.png', img_tensor_np[0, 0, middle_slice])
    cv2.imwrite('img/augmented_seg.png', augmented_seg_np[0, middle_slice]*255)
    cv2.imwrite('img/seg.png', seg_tensor_np[0, middle_slice]*255)

    print(augmentor)