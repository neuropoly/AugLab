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
        apply_to_channel: list[int] = [0],  # Apply to first channel by default
        same_on_batch: bool = False,
        retain_stats: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if kernel_type not in ["Laplace", "Scharr", "GaussianBlur", "UnsharpMask"]:
            raise NotImplementedError('Currently only "Laplace", "Scharr", "GaussianBlur" and "UnsharpMask" are supported.')
        else:
            self.kernel_type = kernel_type
        self.apply_to_channel = apply_to_channel
        self.absolute = kwargs.get('absolute', False)
        self.sigma = kwargs.get('sigma', 1.0)
        self.retain_stats = retain_stats
        # Unsharp mask parameters: amount controls strength of the mask
        self.unsharp_amount = kwargs.get('unsharp_amount', 1.0)

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
            sigma = torch.rand(3, device=device) * self.sigma
            kernel_size = 3
            kernel = get_gaussian_kernel3d(kernel_size, sigma, torch.float32, device)
        elif self.kernel_type == "UnsharpMask":
            # For unsharp masking we use a Gaussian blur kernel; amount is applied in apply_transform.
            sigma = torch.rand(3, device=device) * self.sigma
            kernel_size = 3
            kernel = get_gaussian_kernel3d(kernel_size, sigma, torch.float32, device)
        else:
            raise NotImplementedError('Kernel type not implemented.')
        return kernel
    
    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Initialize kernel
        kernel = self.get_kernel(device=input.device)

        if self.retain_stats:
            # Compute original mean and std for each channel to be processed (per-sample / per-image)
            orig_means = {}
            orig_stds = {}
            for c in self.apply_to_channel:
                x = input[:, c]  # shape [N, ...spatial...]
                reduce_dims = tuple(range(1, x.dim()))
                # store per-sample mean/std (shape [N])
                orig_means[c] = x.mean(dim=reduce_dims)
                orig_stds[c] = x.std(dim=reduce_dims)
        
        # Apply convolution
        for c in self.apply_to_channel:
            if self.kernel_type in ['Laplace', 'GaussianBlur']:
                input[:, c] = apply_convolution(input[:, c], kernel, dim=3)
            elif self.kernel_type == 'UnsharpMask':
                # blur selected channel, compute mask and add scaled mask back Isharp​=I+α(I−G​∗I)
                blurred = apply_convolution(input[:, c], kernel, dim=3)
                mask = input[:, c] - blurred
                input[:, c] = input[:, c] + self.unsharp_amount * mask
            elif self.kernel_type == 'Scharr':
                tot_ = torch.zeros_like(input[:, c], device=input.device)
                for k in kernel:
                    if self.absolute:
                        tot_ += torch.abs(apply_convolution(input[:, c], k, dim=3))
                    else:
                        tot_ += apply_convolution(input[:, c], k, dim=3)
                input[:, c] = tot_
        
        if self.retain_stats:
            # Adjust mean and std to match original
            eps = 1e-8
            for c in self.apply_to_channel:
                x = input[:, c]  # [N, ...]
                reduce_dims = tuple(range(1, x.dim()))
                new_mean = x.mean(dim=reduce_dims)  # [N]
                new_std = x.std(dim=reduce_dims)    # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [x.shape[0]] + [1] * (x.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means[c].view(shape)
                os = orig_stds[c].view(shape)
                input[:, c] = (x - nm) / (ns + eps) * os + om
        return input

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

def get_gaussian_kernel3d(kernel_size: int, sigma: torch.Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    """
    Create a 3D Gaussian kernel by multiplying 1D kernels along each axis.
    Args:
        kernel_size (int)
        sigma (float or tuple of three floats): Standard deviation of the Gaussian kernel.
    """
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor([sigma, sigma, sigma], device=device)
    elif isinstance(sigma, torch.Tensor):
        assert sigma.shape == (3,), "Sigma must be a float or a tensor of three floats."
    else:
        raise TypeError("Sigma must be a float or a tensor of three floats.")

    gz = get_gaussian_kernel1d(kernel_size, sigma[0], dtype, device)
    gy = get_gaussian_kernel1d(kernel_size, sigma[1], dtype, device)
    gx = get_gaussian_kernel1d(kernel_size, sigma[2], dtype, device)

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
        apply_to_channel: list[int] = [0],  # Apply to first channel by default
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.same_on_batch = same_on_batch
        self.apply_to_channel = apply_to_channel
        self.mean = mean
        self.std = std

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Generate Gaussian noise with the same shape as input
        for c in self.apply_to_channel:
            if self.same_on_batch:
                std = torch.rand(1, device=input.device, dtype=input.dtype) * self.std
                noise = torch.randn_like(input[:,c], device=input.device, dtype=input.dtype)
            else:
                std = torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * self.std
                noise = torch.randn_like(input[:,c], device=input.device, dtype=input.dtype)
            noise = noise * std + self.mean
            input[:, c] += noise

        return input

## Multiplicative brightness transform
class RandomBrightnessGPU(ImageOnlyTransform):
    """Apply random brightness adjustment to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        multiplier_range (tuple of float): Range of brightness multipliers. Default is (0.9, 1.1).
        apply_to_channel (list of int): List of channel indices to apply the brightness adjustment to. Default is [0].
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """
    
    def __init__(
        self,
        brightness_range: list[float, float] = (0.9, 1.1),
        apply_to_channel: list[int] = [0],  # Apply to first channel by default
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.brightness_range = brightness_range
        self.apply_to_channel = apply_to_channel

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:

        # Apply brightness adjustment
        for c in self.apply_to_channel:
            if self.same_on_batch:
                factor = torch.rand(1, device=input.device, dtype=input.dtype) * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]
            else:
                factor = torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]
            input[:, c] *= factor
        return input
