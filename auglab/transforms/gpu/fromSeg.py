import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, Optional, Tuple, Union, List, Protocol
from kornia.core import Tensor
import torch.distributed as dist

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
        std_noise_range: list[float] = [0.1, 0.3],
        dilation_iterations_range: list[int] = [1, 3],
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.in_seg = in_seg
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats
        self.std_noise_range = std_noise_range
        self.dilation_iterations_range = dilation_iterations_range

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
                dilation_iterations = torch.randint(self.dilation_iterations_range[0], self.dilation_iterations_range[1]+1, (1,), device=input.device)[0].item()
                for _ in range(dilation_iterations):
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
                std_noise_range = torch.rand(1, device=input.device)[0] * (self.std_noise_range[1] - self.std_noise_range[0]) + self.std_noise_range[0]
                redist_std = torch.maximum(
                    torch.rand(R, device=input.device) * std_noise_range + 0.4 * torch.abs((means - dil_means) * stds / (dil_stds + 1e-6)),
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



class RandomV19ContrastGPU(ImageOnlyTransform):
    """
    AugLab-compatible GPU augmentation that applies V19 Stochastic Semantic
    Decoupling to produce a stochastic guidance map in place of the input.

    Args:
        label_classes: Integer label indices to decouple stochastically.
            Defaults to BraTS convention [1, 2, 3] (NCR, ED, ET).
            Pass your dataset's foreground class indices if they differ.
        num_bins: Histogram bins for the internal DifferentiableHistogram3D.
        p: Probability of applying the transform (standard Kornia convention).
    """

    def __init__(
        self,
        label_classes: Optional[List[int]] = None,
        num_bins: int = 64,
        p: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, **kwargs)
        self._hist = DifferentiableHistogram3D(num_bins=num_bins, value_range=(0.0, 1.0))
        self._generator = V19LabelConditionedTextureGenerator(label_classes=label_classes)

    # ------------------------------------------------------------------
    # Core AugLab contract
    # ------------------------------------------------------------------

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, Any],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input:  Image tensor [B, C, D, H, W], float32, values in [0, 1].
            params: AugLab parameter dict. If a segmentation mask was registered
                    via AugLab's DataKey.MASK / 'seg' entry, it appears here.
                    Supported formats:
                      - One-hot [B, C_seg, D, H, W] with C_seg > 1
                      - Integer index [B, 1, D, H, W] (passed through directly)
            flags:  AugLab flags dict (unused but required by the interface).

        Returns:
            guidance_map: [B, C, D, H, W] stochastic synthesis output, same
                          shape and dtype as input.
        """
        seg_raw: Optional[torch.Tensor] = params.get("seg", None)

        labels: Optional[torch.Tensor] = None
        if seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] > 1:
            # One-hot [B, C_seg, D, H, W] → integer index [B, 1, D, H, W]
            labels = collapse_onehot_to_index(seg_raw)
        elif seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] == 1:
            # Already a single-channel integer index mask — use directly.
            labels = seg_raw.long()

        # normalize image to [0,1]
        data_norm, vmin, vmax = _minmax_norm(input)

        _target_hist, guidance_map, _dup = self._generator(
            input_images=data_norm,
            hist_module=self._hist,
            labels=labels,
        )
        # normalize back with z-score normalization
        guidance_map = _zscore_renorm(_minmax_denorm(guidance_map, vmin, vmax))
        return guidance_map


_SHARED_RNG_COUNTER = 0


def _next_shared_seed() -> int:
    global _SHARED_RNG_COUNTER
    _SHARED_RNG_COUNTER += 1
    seed = (int(torch.initial_seed()) + _SHARED_RNG_COUNTER) % (2**63 - 1)
    if dist.is_available() and dist.is_initialized():
        seed_tensor = torch.tensor([seed], dtype=torch.long)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    return seed


@staticmethod
def _minmax_norm(x: torch.Tensor, eps: float = 1e-8
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample min-max normalise to [0, 1]. Returns (normed, min, max)."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    vmin = x_flat.min(dim=1).values.view(B, 1, 1, 1, 1)
    vmax = x_flat.max(dim=1).values.view(B, 1, 1, 1, 1)
    return (x - vmin) / (vmax - vmin + eps), vmin, vmax

@staticmethod
def _minmax_denorm(x_norm: torch.Tensor, vmin: torch.Tensor,
                    vmax: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x_norm * (vmax - vmin + eps) + vmin

@staticmethod
def _zscore_renorm(x: torch.Tensor, bg_threshold: float = 1e-6) -> torch.Tensor:
    """Per-sample foreground-masked z-score. Mirrors nnUNet's use_mask_for_norm=True.

    Background voxels (abs ≈ 0, zeroed by nnUNet masking) stay at 0.
    Eliminates the train/inference distribution mismatch that would occur
    because nnUNet always z-scores at inference time.
    """
    fg   = x.abs() > bg_threshold
    fg_f = fg.float()
    n    = fg_f.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1)
    mean = (x * fg_f).sum(dim=(2, 3, 4), keepdim=True) / n
    var  = ((x - mean).pow(2) * fg_f).sum(dim=(2, 3, 4), keepdim=True) / n
    std  = var.sqrt().clamp(min=1e-8)
    return torch.where(fg, (x - mean) / std, torch.zeros_like(x))

def _shared_cpu_generator() -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_next_shared_seed())
    return generator

def _shared_rand(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.rand(shape, device=device, dtype=dtype)
    rand_cpu = torch.rand(shape, generator=_shared_cpu_generator(), device="cpu", dtype=dtype)
    return rand_cpu.to(device=device, dtype=dtype)


def collapse_onehot_to_index(seg_raw: torch.Tensor) -> torch.Tensor:
    """
    Convert a one-hot segmentation mask to a single-channel integer index mask.

    Args:
        seg_raw: One-hot tensor [B, C_seg, D, H, W], bool or float.
                 Channel 0 is assumed to be *absent* (background is implicit).
                 Each foreground channel c encodes class index (c + 1).

    Returns:
        labels: Integer index tensor [B, 1, D, H, W].
                Background voxels (all-zero across channels) map to 0.
                Foreground voxels map to argmax(seg_raw, dim=1) + 1.
    """
    foreground_mask = seg_raw.any(dim=1, keepdim=True)               # [B,1,D,H,W] bool
    labels = torch.argmax(seg_raw, dim=1, keepdim=True).long() + 1   # 0-based → 1-based
    labels = torch.where(foreground_mask, labels, torch.zeros_like(labels))
    return labels

class HistogramModuleLike(Protocol):
    num_bins: int
    min_value: float
    max_value: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

class BaseTargetGenerator(nn.Module):
    """Strategy interface for guidance and target histogram generation."""

    def forward(
        self,
        input_images: torch.Tensor,
        num_bins: int,
        num_chunks: int,
        dark_threshold: float,
        hist_module: HistogramModuleLike,
        return_guidance_map: bool = True,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class V19LabelConditionedTextureGenerator(BaseTargetGenerator):
    """
    V19 Stochastic Semantic Decoupling: Merges geometric label-priors with
    texture-preserving latent space.
    """

    def __init__(self, label_classes: Optional[List[int]] = None):
        super().__init__()
        self.label_classes: List[int] = label_classes if label_classes is not None else [1, 2, 3]

    def __call__(
        self,
        input_images: torch.Tensor,
        hist_module: nn.Module,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = input_images
        B, C, D, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Step A: Base v18_6 Background Synthesis
        mask = images > 0.01

        y = images.clone()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
            images_f = images.float()

            mu_base = _shared_rand((B, 8), device=device, dtype=torch.float32)
            alpha_base = _shared_rand((B, 8), device=device, dtype=torch.float32) * 1.5 + 0.5

            q_edges = torch.linspace(0, 1, 9, device=device)

            c_i = torch.bucketize(images_f, q_edges) - 1
            c_i = torch.clamp(c_i, 0, 7)

            mu_c = mu_base.view(B, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            alpha_c = alpha_base.view(B, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            # v19_c bias fix: center the affine shift on the chunk midpoint rather than
            # the lower edge, so alpha × offset is symmetric around 0 within each chunk.
            q_c_lower = q_edges[:-1].view(1, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            q_c_upper = q_edges[1:].view(1, 8, 1, 1, 1).expand(B, 8, D, H, W).gather(1, c_i)
            q_c_center = (q_c_lower + q_c_upper) * 0.5

            y_base = mu_c + alpha_c * (images_f - q_c_center)
            y = torch.where(mask, y_base.to(dtype), y)

        # Step B: Stochastic Semantic Decoupling
        if labels is not None:
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
                if labels.dim() == 4:
                    labels = labels.unsqueeze(1)
                if any(dim <= 0 for dim in labels.shape[2:]):
                    labels = None
                else:
                    if labels.shape[2:] != images.shape[2:]:
                        labels = F.interpolate(labels.float(), size=images.shape[2:], mode="nearest")
                    labels = labels.to(device=device)
                    y_f = y.float()
                    images_f = images.float()

                    for c in self.label_classes:
                        decouple = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32) > 0.5

                        mu_path = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32)
                        alpha_path = _shared_rand((B, 1, 1, 1, 1), device=device, dtype=torch.float32) * 1.5 + 0.5

                        class_mask = (labels == c)

                        class_sum = (images_f * class_mask).sum(dim=(1, 2, 3, 4), keepdim=True)
                        class_count = class_mask.sum(dim=(1, 2, 3, 4), keepdim=True)
                        class_count_safe = torch.clamp(class_count, min=1.0)
                        mean_c = class_sum / class_count_safe

                        y_override = mu_path + alpha_path * (images_f - mean_c)

                        valid_override = class_mask & decouple & (class_count > 0)
                        y_f = torch.where(valid_override, y_override, y_f)

                    y = y_f.to(dtype)

        # Step C: Masking & Clamping
        y = torch.clamp(y, 0.0, 1.0)
        y = torch.where(mask, y, torch.zeros_like(y))

        target_hist = hist_module(y)
        return target_hist, y, y


class DifferentiableHistogram3D(nn.Module):
    """Differentiable soft histogram for 3D volumes returning RAW VOXEL COUNTS."""

    def __init__(self, num_bins: int = 64, value_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.min_value = float(value_range[0])
        self.max_value = float(value_range[1])
        self.eps = eps

        bin_centers = torch.linspace(self.min_value, self.max_value, num_bins)
        self.register_buffer("bin_centers", bin_centers.view(1, 1, num_bins, 1), persistent=False)
        self.bin_width = (self.max_value - self.min_value) / max(num_bins - 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")

        b, c, *_ = x.shape
        flat_x = x.reshape(b, c, -1)

        scaled = (flat_x - self.min_value) / (self.bin_width + self.eps)
        left_idx = torch.floor(scaled).to(torch.long)
        right_idx = left_idx + 1

        wl = (right_idx.to(flat_x.dtype) - scaled).clamp(0.0, 1.0)
        wr = (scaled - left_idx.to(flat_x.dtype)).clamp(0.0, 1.0)

        left_idx = left_idx.clamp(0, self.num_bins - 1)
        right_idx = right_idx.clamp(0, self.num_bins - 1)

        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError("Mask shape must match the input tensor shape.")
            flat_mask = mask.reshape(b, c, -1).to(dtype=flat_x.dtype)
            wl = wl * flat_mask
            wr = wr * flat_mask

        hist = torch.zeros((b, c, self.num_bins), device=x.device, dtype=x.dtype)
        hist.scatter_add_(2, left_idx, wl)
        hist.scatter_add_(2, right_idx, wr)

        return hist
