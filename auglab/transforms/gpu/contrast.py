import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms._functional_tensor as F_t

from typing import Any, Dict, Optional
from kornia.core import Tensor

from auglab.transforms.gpu.base import ImageOnlyTransform

## Convolution transform
class RandomConvTransformGPU(ImageOnlyTransform):
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
    def __init__(
        self,
        kernel_type: str = 'Laplace', 
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if kernel_type not in  ["Laplace","Scharr","GaussianBlur"]:
            raise NotImplementedError('Currently only "Laplace", "Scharr" and "GaussianBlur" are supported.')
        else:
            self.kernel_type = kernel_type
        self.absolute = kwargs.get('absolute', False)
        self.sigma = kwargs.get('sigma', 1.0)

    def get_kernel(self, device: torch.device) -> torch.Tensor:
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
        elif self.kernel_type == "GaussianBlur":
            sigma = torch.rand((1, 3), device=device) * self.sigma
            kernel_size = 3
            kernel = get_gaussian_kernel3d(kernel_size, sigma, torch.float32, device)
        else:
            raise NotImplementedError('Currently only "Laplace" and "Scharr" are supported.')
        return kernel
    
    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Initialize kernel
        kernel = self.get_kernel(device=input.device)
        
        # Apply convolution
        if self.kernel_type in ['Laplace', 'GaussianBlur']:
            tot_ = apply_convolution(input, kernel, dim=3)
        elif self.kernel_type == 'Scharr':
            tot_ = torch.zeros_like(input, device=input.device)
            for k in kernel:
                if self.absolute:
                    tot_ += torch.abs(apply_convolution(input, k, dim=3))
                else:
                    tot_ += apply_convolution(input, k, dim=3)
        return tot_

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

def get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Create a 1D Gaussian kernel."""

    x = torch.arange(kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def get_gaussian_kernel3d(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    """
    Create a 3D Gaussian kernel by multiplying 1D kernels along each axis.
    Args:
        kernel_size (int)
        sigma (float or tuple of three floats): Standard deviation of the Gaussian kernel.
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)
    elif isinstance(sigma, (list, tuple)):
        assert len(sigma) == 3, "Sigma must be a float or a tuple of three floats."
    else:
        raise TypeError("Sigma must be a float or a tuple of three floats.")
    
    gz = get_gaussian_kernel1d(kernel_size, sigma[0], device, dtype)
    gy = get_gaussian_kernel1d(kernel_size, sigma[1], device, dtype)
    gx = get_gaussian_kernel1d(kernel_size, sigma[2], device, dtype)

    # Outer product using broadcasting
    kernel = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]

    # Normalize
    kernel /= kernel.sum()

    return kernel

## Noise transform
class RandomGaussianNoiseGPU(ImageOnlyTransform):
    """Add random Gaussian noise to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        mean (float): Mean of the Gaussian noise. Default is 0.0.
        std (float): Standard deviation of the Gaussian noise. Default is 0.1.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with added Gaussian noise.
    """
    
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.1,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.mean = mean
        self.std = std

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Generate Gaussian noise with the same shape as input
        std = torch.rand(1, device=input.device, dtype=input.dtype) * self.std
        noise = torch.randn_like(input, device=input.device, dtype=input.dtype)
        noise = noise * std + self.mean
        
        # Add noise to the input
        output = input + noise
        
        return output
