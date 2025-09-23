import os, json

from kornia.augmentation import RandomAffine3D
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample
from kornia.core import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms._functional_tensor as F_t
from torch.distributions import Bernoulli

from typing import Any, Dict, Optional

class DataAugmentationGPU(AugmentationSequential):
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
        conv_params = self.transform_params.get('ConvTransform')
        transforms.append(RandomTransformGPU(
            ConvTransformGPU(kernel_type=conv_params['kernel_type'], spatial_dims=3, absolute=conv_params['absolute']),
            apply_probability=conv_params.get('probability')
        ))

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
        transforms.append(RandomAffine3D(
            degrees=affine_params.get('degrees'),
            translate=affine_params.get('translate'),
            scale=affine_params.get('scale'),
            shears=None, #affine_params.get('shear'),
            resample='bilinear',
            p=affine_params.get('probability'),
            keepdim=False
        ))
        return transforms

class RandomAffine3DCustom(RandomAffine3D):
    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.

        """
        resample_method: Optional[Resample]
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        output = self.apply_transform(input, params, flags, transform)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output

class RandomTransformGPU(nn.Module):
    def __init__(self, transform: nn.Module, apply_probability: float = 1):
        super().__init__()
        self.transform = transform
        self.apply_probability = apply_probability

    def apply_transform(self, batch_size) -> bool:
        return Bernoulli(self.apply_probability).sample(torch.Size([batch_size]))

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.apply_transform(x.shape[0]).to(dtype=torch.bool, device=x.device)
        if mask.all():
            return self.transform(x)
        elif not mask.any():
            return x
        else:
            # Apply transform only to selected images in the batch
            x_out = x.clone()
            x_out[mask] = self.transform(x[mask])
            return x_out

class ConvTransformGPU(nn.Module):
    """Apply convolution to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.
    Based on https://docs.pytorch.org/vision/0.9/transforms.html#torchvision.transforms.GaussianBlur

    Args:
        kernel_type (str): Type of convolution kernel, either 'Laplace' or 'Scharr'. Default is 'Laplace'.
        spatial_dims (int): Number of spatial dimensions of the input image, either 2 or 3. Default is 2.
        absolute (bool): If True, take the absolute value of the convolution result. Default is False.
        retain_stats (bool): If True, retain the original mean and standard deviation of the image after convolution. Default is False.

    Returns:
        Tensor: Convolved version of the input image.

    """
    def __init__(self, kernel_type: str = 'Laplace', spatial_dims: int = 3, absolute: bool = False):
        super().__init__()
        if kernel_type not in  ["Laplace","Scharr"]:
            raise NotImplementedError('Currently only "Laplace" and "Scharr" are supported.')
        else:
            self.kernel_type = kernel_type
        self.absolute = absolute
        self.spatial_dims = spatial_dims
        self.kernel = None

    def get_kernel(self, device: torch.device) -> torch.Tensor:
        spatial_dims = self.spatial_dims
        if spatial_dims == 2:
            if self.kernel_type == "Laplace":
                kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=device)
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, -10], [-3, 0, 3]], dtype=torch.float32, device=device)
                kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32, device=device)
                kernel = [kernel_x, kernel_y]
        elif spatial_dims == 3:
            if self.kernel_type == "Laplace":
                kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32, device=device)
                kernel[1, 1, 1] = 26.0
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[[  9,    0,    -9],
                                          [ 30,    0,   -30],
                                          [  9,    0,    -9]],

                                          [[ 30,    0,   -30],
                                           [100,    0,  -100],
                                           [ 30,    0,   -30]],

                                          [[  9,    0,    -9],
                                           [ 30,    0,   -30],
                                           [  9,    0,    -9]]], dtype=torch.float32, device=device)
                
                kernel_y = torch.tensor([[[    9,   30,    9],
                                          [    0,    0,    0],
                                          [   -9,  -30,   -9]],

                                         [[  30,  100,   30],
                                          [   0,    0,    0],
                                          [ -30, -100,  -30]],

                                         [[   9,   30,    9],
                                          [   0,    0,    0],
                                          [  -9,  -30,   -9]]], dtype=torch.float32, device=device)

                kernel_z = torch.tensor([[[   9,   30,   9],
                                          [  30,  100,  30],
                                          [   9,   30,   9]],

                                         [[   0,    0,   0],
                                          [   0,    0,   0],
                                          [   0,    0,   0]],

                                         [[   -9,  -30,  -9],
                                          [  -30, -100, -30],
                                          [   -9,  -30,  -9]]], dtype=torch.float32, device=device)
                kernel = [kernel_x, kernel_y, kernel_z]
        else:
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")
        return kernel

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        '''
        We expect (N, C, X, Y) or (N, C, X, Y, Z) shaped inputs for image and seg
        '''
        # Initialize kernel
        self.kernel = self.get_kernel(device=img.device)

        # Apply convolution
        img = apply_convolution(img, self.kernel, self.spatial_dims)

        # Apply absolute value if specified
        if self.absolute:
            img = torch.abs(img)
        return img

def apply_convolution(img: torch.Tensor, kernel: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    Based on https://github.com/pytorch/vision/blob/e3b5d3a8bf5e8636462fd8bce9897bccc690b2a0/torchvision/transforms/_functional_tensor.py#L746
    '''
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    if dim == 2:
        kernel = kernel.expand(img.shape[-(1 + dim)], 1, kernel.shape[0], kernel.shape[1])
        padding = [kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2]
    elif dim == 3:
        kernel = kernel.expand(img.shape[-(1 + dim)], 1, kernel.shape[0], kernel.shape[1], kernel.shape[2])
        padding = [kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2] + [kernel.shape[4] // 2, kernel.shape[4] // 2]
    else:
        raise ValueError(f"Only 2D and 3D convolution are supported. Got {dim}D.")
    
    img, need_cast, need_squeeze, out_dtype = F_t._cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    img = F.pad(img, padding, mode="reflect")
    if dim == 2:
        img = F.conv2d(img, kernel, groups=img.shape[-(1 + dim)])
    else:  # dim == 3
        img = F.conv3d(img, kernel, groups=img.shape[-(1 + dim)])

    img = F_t._cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img
