import torch
from torch.nn import functional as F

from typing import Any, Dict, Optional, Tuple, Union, List
from kornia.core import Tensor

from auglab.transforms.gpu.base import ImageOnlyTransform


def _binary_dilation(mask: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """Binary dilation using max-pooling. Supports 2D/3D masks shaped (..., H, W) or (D, H, W).
    Expects mask of dtype bool or 0/1 float.
    """
    is_3d = (mask.dim() == 3)
    kernel_size = 3
    pad = 1
    x = mask.float()
    if is_3d:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        for _ in range(iterations):
            x = F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=pad)
        x = x.squeeze(0).squeeze(0)
    else:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        for _ in range(iterations):
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        x = x.squeeze(0).squeeze(0)
    return (x > 0).to(mask.dtype)


def _normal_pdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    inv = 1.0 / (std + 1e-6)
    return (inv / (torch.sqrt(torch.tensor(2.0 * 3.141592653589793, device=x.device, dtype=x.dtype)))) * torch.exp(
        -0.5 * ((x - mean) * inv) ** 2
    )


## Redistribute segmentation values transform (GPU)
class RandomRedistributeSegGPU(ImageOnlyTransform):
    """Redistribute image values using segmentation regions (GPU version).

    Mirrors the CPU `RedistributeTransform` behavior using GPU-friendly ops.
    Works with inputs shaped [N, C, H, W] or [N, C, D, H, W].
    """

    def __init__(
        self,
        in_seg: float = 0.2,
        apply_to_channel: list[int] = [0],
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

    @torch.no_grad()
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Expect segmentation provided in params: shape [N, 1, ...] or [N, C_seg, ...]
        if 'seg' not in params:
            return input
        seg = params['seg']
        if seg.dim() != input.dim():
            # Allow seg [N, ...] by adding channel dim
            if seg.dim() == input.dim() - 1:
                seg = seg.unsqueeze(1)
            else:
                return input

        spatial_dims = input.dim() - 2
        if spatial_dims not in (2, 3):
            return input

        N = input.shape[0]

        # Apply per selected image channel and per batch sample
        for c in self.apply_to_channel:
            # process batch elements independently
            for b in range(N):
                img = input[b, c]
                seg_b = seg[b]

                # Optionally retain original stats
                if self.retain_stats:
                    orig_mean = img.mean()
                    orig_std = img.std()

                # Normalize image to [0,1]
                img_min, img_max = img.min(), img.max()
                x = (img - img_min) / (img_max - img_min + 1e-6)

                to_add = torch.zeros_like(x)

                # Decide whether to add only inside segmentation regions
                in_seg_bool = (1 - torch.rand(1, device=input.device)) <= self.in_seg

                for l_mask in seg_b:
                    l_mask = l_mask.bool()
                    if l_mask.any():
                        l_vals = x[l_mask]
                        l_mean = l_vals.mean()
                        l_std = l_vals.std()

                        # Dilate mask 3 iterations with 3x3x3 structuring element equivalent
                        l_mask_dilate = _binary_dilation(l_mask, iterations=3)
                        l_mask_dilate_excl = l_mask_dilate & (~l_mask)

                        if l_mask_dilate_excl.any():
                            dl_vals = x[l_mask_dilate_excl]
                            l_mean_dilate = dl_vals.mean()
                            l_std_dilate = dl_vals.std()
                        else:
                            l_mean_dilate, l_std_dilate = l_mean, l_std

                        redist_std = torch.maximum(
                            torch.rand(1, device=input.device) * 0.2
                            + 0.4 * torch.abs((l_mean - l_mean_dilate) * l_std / (l_std_dilate + 1e-6)),
                            torch.tensor([0.01], device=input.device, dtype=input.dtype),
                        )

                        # Build additive term using normal pdf centered at l_mean
                        if in_seg_bool:
                            vals = x[l_mask]
                            to_add[l_mask] += _normal_pdf(vals, l_mean, redist_std) * (2 * torch.rand(1, device=input.device) - 1)
                        else:
                            to_add += _normal_pdf(x, l_mean, redist_std) * (2 * torch.rand(1, device=input.device) - 1)

                # Normalize to_add and apply
                tmin, tmax = to_add.min(), to_add.max()
                x = x + 2 * (to_add - tmin) / (tmax - tmin + 1e-6)

                # Restore original stats if requested
                if self.retain_stats:
                    mean = x.mean()
                    std = x.std()
                    x = (x - mean) / torch.clamp(std, min=1e-7)
                    x = x * orig_std + orig_mean

                # Write back to input
                input[b, c] = x

        return input
