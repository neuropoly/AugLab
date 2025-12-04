import torch
from torch.nn import functional as F

from typing import Any, Dict, Optional, Tuple, Union, List
from kornia.core import Tensor

from auglab.transforms.gpu.base import ImageOnlyTransform


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
            img_batch = input[:, c]  # (N, [...])
            # Sanitize incoming values to prevent NaN/Inf propagation
            img_batch = torch.nan_to_num(img_batch, nan=0.0, posinf=0.0, neginf=0.0)

            # Optionally retain original stats (vectorized per sample)
            if self.retain_stats:
                flat = img_batch.view(N, -1)
                orig_mean = flat.mean(dim=1)
                # Use unbiased=False to avoid NaNs for tiny tensors
                orig_std = flat.std(dim=1, unbiased=False)

            # Normalize entire batch to [0,1] per sample
            img_min = img_batch.view(N, -1).min(dim=1)[0].view(N, *([1] * (img_batch.dim()-1)))
            img_max = img_batch.view(N, -1).max(dim=1)[0].view(N, *([1] * (img_batch.dim()-1)))
            denom = (img_max - img_min).clamp_min(1e-6)
            x_batch = (img_batch - img_min) / denom

            # Iterate per sample (seg can differ in shape or labels per sample)
            for b in range(N):
                x = x_batch[b]
                seg_b = seg[b]  # shape (R, ...)

                # Quick skip if no foreground
                if (seg_b > 0).sum() == 0:
                    input[b, c] = x
                    continue

                # Decide redistribution mode once per sample
                # Scalar random flag for redistribution mode
                in_seg_bool = torch.rand((), device=input.device) <= self.in_seg

                # Binary masks for regions
                masks = seg_b.bool()  # (R, ...)
                R = masks.shape[0]

                # Vectorized dilation for all regions (3 iterations)
                dilated = masks.float()
                for _ in range(3):
                    if spatial_dims == 3:
                        dilated = F.max_pool3d(dilated.unsqueeze(0), 3, 1, 1).squeeze(0)
                    else:
                        dilated = F.max_pool2d(dilated.unsqueeze(0), 3, 1, 1).squeeze(0)
                dilated_excl = (dilated > 0) & (~masks)

                # Flatten for stats
                x_flat = x.view(1, -1)  # (1, S)
                mask_flat = masks.view(R, -1)
                dil_flat = dilated_excl.view(R, -1)

                # Region counts
                counts = mask_flat.sum(dim=1).clamp_min(1)
                # Means
                means = (mask_flat * x_flat).sum(dim=1) / counts
                # Std (compute variance then sqrt) avoid indexing overhead
                diffs = (x_flat - means.view(R,1)) * mask_flat
                vars = (diffs * diffs).sum(dim=1) / counts.clamp_min(1)
                stds = vars.sqrt()

                # Dilated stats
                dil_counts = dil_flat.sum(dim=1).clamp_min(1)
                dil_means = (dil_flat * x_flat).sum(dim=1) / dil_counts
                dil_diffs = (x_flat - dil_means.view(R,1)) * dil_flat
                dil_vars = (dil_diffs * dil_diffs).sum(dim=1) / dil_counts
                dil_stds = dil_vars.sqrt()

                # redist_std per region
                redist_std = torch.maximum(
                    torch.rand(R, device=input.device) * 0.2 + 0.4 * torch.abs((means - dil_means) * stds / (dil_stds + 1e-6)),
                    torch.full((R,), 0.01, device=input.device, dtype=input.dtype)
                )

                # Build additive term
                to_add = torch.zeros_like(x)
                rand_sign = (2 * torch.rand(R, device=input.device) - 1)  # random sign factor per region
                if in_seg_bool.item():
                    # Only inside region
                    for r in range(R):
                        if counts[r] == 0:  # skip empty
                            continue
                        region_vals = x[mask_flat[r].view(x.shape)]
                        pdf_vals = _normal_pdf(region_vals, means[r], redist_std[r]) * rand_sign[r]
                        to_add[mask_flat[r].view(x.shape)] += pdf_vals
                else:
                    # Global additive influence per region
                    pdf_all = []
                    for r in range(R):
                        if counts[r] == 0:
                            continue
                        pdf_all.append(_normal_pdf(x, means[r], redist_std[r]) * rand_sign[r])
                    if pdf_all:
                        to_add += torch.stack(pdf_all, dim=0).sum(dim=0)

                # Normalize to_add if non-zero
                tmin, tmax = to_add.min(), to_add.max()
                if (tmax - tmin) > 1e-8:
                    x = x + 2 * (to_add - tmin) / (tmax - tmin + 1e-6)

                # Restore stats
                if self.retain_stats:
                    mean = x.mean()
                    # Use unbiased=False to avoid NaNs on degenerate shapes
                    std = x.std(unbiased=False)
                    x = (x - mean) / torch.clamp(std, min=1e-7)
                    x = x * orig_std[b] + orig_mean[b]

                # Final safety: check if nan/inf appeared
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning nan: {self.__class__.__name__}", flush=True)
                    continue
                input[b, c] = x

        return input
