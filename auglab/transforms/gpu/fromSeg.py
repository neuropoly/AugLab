import torch

from typing import Any, Dict, Optional
from kornia.core import Tensor

from auglab.transforms.gpu.base import ImageOnlyTransform
from typing import Any, Dict, Optional, Tuple, Union, List


## Redistribute segmentation values transform
class RandomRedistributeSegGPU(ImageOnlyTransform):
    """Redistribute segmentation values.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        in_seg (float): Segmentation redistribution parameter. Default is 0.2.
        apply_to_channel (list of int): List of channel indices to apply the gamma adjustment to. Default is [0].
        retain_stats (bool): If True, retain the original mean and standard deviation of the image after gamma adjustment. Default is False.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """
    
    def __init__(
        self,
        in_seg: float = 0.2,
        apply_to_channel: list[int] = [0],  # Apply to first channel by default
        retain_stats: bool = False,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.in_seg = in_seg
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        
        # Load segmentation from params
        seg = params['seg']

        # Apply brightness adjustment
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            if self.retain_stats:
                reduce_dims = tuple(range(1, channel_data.dim()))
                # store per-sample mean/std (shape [N])
                orig_means = channel_data.mean(dim=reduce_dims)
                orig_stds = channel_data.std(dim=reduce_dims)
            
            if self.same_on_batch:
                factor = torch.rand(1, device=input.device, dtype=input.dtype) * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]
                for i in range(input.shape[0]):
                    mean = channel_data[i].mean()
                    channel_data[i] -= mean
                    channel_data[i] *= factor
                    channel_data[i] += mean
            else:
                factor = torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]
                for i in range(input.shape[0]):
                    mean = channel_data[i].mean()
                    channel_data[i] -= mean
                    channel_data[i] *= factor[i]
                    channel_data[i] += mean
            
            if self.retain_stats:
                # Adjust mean and std to match original
                eps = 1e-8
                reduce_dims = tuple(range(1, channel_data.dim()))
                new_mean = channel_data.mean(dim=reduce_dims)  # [N]
                new_std = channel_data.std(dim=reduce_dims)    # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [channel_data.shape[0]] + [1] * (channel_data.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means.view(shape)
                os = orig_stds.view(shape)
                input[:, c] = (channel_data - nm) / (ns + eps) * os + om
            else:
                input[:, c] = channel_data

        return input