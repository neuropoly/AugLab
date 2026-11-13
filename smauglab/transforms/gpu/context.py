"""What lies around the scan: shift the scan inside the patch, and synthesise what is outside it.

A sagittal spine slab is ~8 cm wide. nnU-Net centres it in a 128 mm patch and pads the rest
with 0 (seg -1 -> 0), so the network sees spine in every training window and nothing but a
flat zero slab beyond +-4 cm. On a whole-body axial scan it then finds vertebrae in muscle,
gluteals, fat and bowel: on NAKO VIBE most of its false-positive volume lies more than 5 cm
from the spine. `RandomFOVExtensionGPU` shows it windows where the spine sits at the edge or
is absent, surrounded by tissue that is labelled background.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from kornia.augmentation._3d.base import RigidAffineAugmentationBase3D
from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling
from torch import Tensor

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import segmentation_from
from smauglab.transforms.gpu.contrast import _foreground
from smauglab.transforms.synthseg.functional import _quadratic_log_likelihood_terms, gaussian_blur_3d, random_svf_field, warp_volume

# How the region map is carried into the empty part of the patch. Constants rather than
# parameters: they shape a texture that is always labelled background, and the screen that
# introduced the transform varied `shift` and `p`, not these.
_SCALE_RANGE = (0.75, 1.33)  # log-uniform, ~nnU-Net's 0.7-1.4
_EM_ITERS = 20  # SynthSeg's em_n_iters default
_EM_SAMPLES = 100_000  # SynthSeg's em_max_fit_voxels default
_SEAM = 1  # voxels of interpolated padding edge never used as a source
_CLOSE_ITERS = 8  # max-pool steps that close what the warped layers missed
_TEXTURE_SIGMA = 1.0  # correlation length of the within-region texture, in voxels
_RENDER_SIGMA = 0.5  # SynthSeg's blur at the atlas resolution, in voxels
_RENDER_BLUR_RANGE = 1.15  # SynthSeg's DynamicGaussianBlur jitter


def _dilate(mask: Tensor, radius: int) -> Tensor:
    """Binary dilation of a `[N, 1, D, H, W]` bool mask by a cube of half-width `radius`.

    Three 1-D max-pools rather than one dense `(2r+1)^3` pool: the same result for a cube.
    """
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    x = mask.float()
    for kernel in ((k, 1, 1), (1, k, 1), (1, 1, k)):
        x = F.max_pool3d(x, kernel, stride=1, padding=tuple(s // 2 for s in kernel))
    return x > 0


def _shift(volume: Tensor, offsets: Tensor | Sequence[Sequence[int]], fill: float | bool) -> Tensor:
    """Translate each row of `[N, C, D, H, W]` by whole voxels; vacated voxels get `fill`.

    `offsets[b]` is the `(d, h, w)` displacement of row `b`, as an `[N, 3]` integer tensor
    or nested lists. Integer and exact, so the image and the mask move identically and no
    label is interpolated. One gather per axis on the volume's device, so a device tensor
    of offsets costs no host read.
    """
    offsets = torch.as_tensor(offsets, dtype=torch.long, device=volume.device).view(-1, 3)
    n = volume.shape[0]
    out = volume
    for axis in range(3):
        size = volume.shape[2 + axis]
        view = [n, 1, 1, 1, 1]
        view[2 + axis] = size
        src = torch.arange(size, device=volume.device).view(1, -1) - offsets[:, axis : axis + 1]  # [N, size]
        valid = ((src >= 0) & (src < size)).view(view)
        index = src.clamp(0, size - 1).view(view).expand_as(out)
        out = torch.where(valid, out.gather(2 + axis, index), fill)
    return out


def _touched_faces(labelled: Tensor, tol: int) -> Tensor:
    """`[N, 3, 2]` bool: whether each row's labels come within `tol` voxels of the low and the high face of each axis.

    `labelled` is `[N, 1, D, H, W]` bool. Per-axis profiles from `amax`, so no host read.
    """
    faces = []
    for axis in range(3):
        others = tuple(2 + a for a in range(3) if a != axis)
        profile = labelled.float().amax(dim=(1, *others)) > 0  # [N, size]
        k = min(tol + 1, profile.shape[1])
        faces.append(torch.stack([profile[:, :k].any(dim=1), profile[:, -k:].any(dim=1)], dim=1))
    return torch.stack(faces, dim=1)


def _em_regions(x: Tensor, weight: Tensor, n_regions: Tensor, k_max: int, eps: float = 1e-6) -> Tensor:
    """Fit a 1-D Gaussian mixture per row by EM and return each voxel's most likely component.

    The batched, weighted form of `synthseg.functional._em_gmm_1d`: every row is fitted at
    once, on `_EM_SAMPLES` voxels drawn uniformly and weighted by `weight`, so selecting the
    background costs no boolean indexing and no host synchronisation. Row `b` uses its first
    `n_regions[b]` of `k_max` components; the rest are masked out of the softmax.

    Args:
        x: `[N, V]` intensities, all voxels of the patch.
        weight: `[N, V]` 1.0 where a voxel may be fitted, else 0.0.
        n_regions: `[N]` long, components per row, at least 1.
        k_max: number of component slots.

    Returns:
        `[N, V]` long, the component index of every voxel.
    """
    n, v = x.shape
    idx = torch.randint(0, v, (n, _EM_SAMPLES), device=x.device)
    xs, ws = x.gather(1, idx), weight.gather(1, idx)

    comp = torch.arange(k_max, device=x.device)
    active = comp.view(1, -1) < n_regions.view(-1, 1)  # [N, K]
    total = ws.sum(1, keepdim=True).clamp_min(eps)
    shift = (ws * xs).sum(1, keepdim=True) / total
    centred = xs - shift

    big = torch.finfo(x.dtype).max
    lo = torch.where(ws > 0, centred, big).amin(1, keepdim=True)
    hi = torch.where(ws > 0, centred, -big).amax(1, keepdim=True)
    lo, hi = torch.where(total > eps, lo, 0.0), torch.where(total > eps, hi, 0.0)
    means = lo + (hi - lo) / (n_regions.view(-1, 1) - 1).clamp_min(1) * comp.view(1, -1)
    var = ((ws * centred * centred).sum(1, keepdim=True) / total).clamp_min(eps).expand(n, k_max)
    mix = active.float() / n_regions.view(-1, 1)

    design = torch.stack([centred * centred, centred, torch.ones_like(centred)], dim=2)  # [N, S, 3]
    for _ in range(_EM_ITERS):
        terms = _quadratic_log_likelihood_terms(means, var, mix, eps).permute(1, 0, 2)  # [N, 3, K]
        resp = torch.softmax((design @ terms).masked_fill(~active.unsqueeze(1), -math.inf), dim=2) * ws.unsqueeze(2)
        # Sums of r*x^2, r*x, r per component. A broadcast product rather than
        # `resp.mT @ design`: cuBLAS takes ~7.5 ms for that reduction shape, this ~0.04 ms.
        moments = (resp.unsqueeze(3) * design.unsqueeze(2)).sum(1)  # [N, K, 3]
        nk = moments[..., 2].clamp_min(eps)
        mix = torch.where(active, nk / total, 0.0)
        means = moments[..., 1] / nk
        var = (moments[..., 0] / nk - means * means).clamp_min(eps)

    terms = _quadratic_log_likelihood_terms(means, var, mix, eps).permute(1, 0, 2)
    c = (x - shift).unsqueeze(2)
    logits = torch.cat([c * c, c, torch.ones_like(c)], dim=2) @ terms  # [N, V, K]
    return logits.masked_fill(~active.unsqueeze(1), -math.inf).argmax(dim=2)


def _random_similarity(n: int, device: torch.device) -> Tensor:
    """`[n, 3, 3]` random rotation (uniform axis, angle in [-pi, pi]) x log-uniform scale x mirror."""
    axis = F.normalize(torch.randn(n, 3, device=device), dim=1)
    angle = (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi
    kx = torch.zeros(n, 3, 3, device=device)
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    kx = kx - kx.transpose(1, 2)
    eye = torch.eye(3, device=device).expand(n, 3, 3)
    sin, cos = angle.sin().view(n, 1, 1), angle.cos().view(n, 1, 1)
    rotation = eye + sin * kx + (1.0 - cos) * (kx @ kx)  # Rodrigues
    lo, hi = math.log(_SCALE_RANGE[0]), math.log(_SCALE_RANGE[1])
    scale = torch.exp(lo + torch.rand(n, 1, device=device) * (hi - lo))
    mirror = torch.where(torch.rand(n, 3, device=device) < 0.5, -1.0, 1.0)
    return rotation @ torch.diag_embed(scale * mirror)


def _extend_regions(source: Tensor, seed: Tensor, todo: Tensor, layers: int, close_iters: int = _CLOSE_ITERS, default: int = 0) -> Tensor:
    """Write region labels into `todo` from warped copies of a fixed source map.

    `source` and `seed` are `[N, 1, D, H, W]` long, `-1` where unlabelled; `seed` is the
    map the result starts from. Each layer warps `source` by a random rotation, scale,
    mirror and translation with reflection padding, and keeps the voxels of `todo` it
    lands a label on. It never warps what earlier layers wrote: copying copies lets
    whichever region an early layer spread take over the fill. Whatever is still
    unlabelled after the layers takes a neighbour's label by up to `close_iters` max-pool
    steps, then `default`.
    """
    n = source.shape[0]
    shape = source.shape[2:]
    half = torch.tensor(shape, device=source.device, dtype=torch.float32).view(1, 3, 1) / 2.0
    cur = seed.clone()
    for _ in range(layers):
        affine = torch.zeros(n, 4, 4, device=source.device)
        affine[:, :3, :3] = _random_similarity(n, source.device)
        affine[:, :3, 3:] = (torch.rand(n, 3, 1, device=source.device) * 2.0 - 1.0) * half
        affine[:, 3, 3] = 1.0
        warped = warp_volume((source + 1).float(), affine=affine, interp="nearest", padding_mode="reflection").round().long() - 1
        take = todo & (warped >= 0)
        cur = torch.where(take, warped, cur)
        todo = todo & ~take
    for _ in range(close_iters):
        grown = F.max_pool3d((cur + 1).float(), 3, stride=1, padding=1).long() - 1
        take = todo & (grown >= 0)
        cur = torch.where(take, grown, cur)
        todo = todo & ~take
    return torch.where(todo, default, cur)


def _render(image: Tensor, atlas: Tensor, layout: Tensor, k_max: int, eps: float = 1e-6) -> Tensor:
    """Draw every voxel of `layout` from its region's Gaussian, per channel, then blur as SynthSeg does.

    Means and variances are measured where `atlas` labels `image` (`-1` elsewhere), the
    patch before the shift, so the fill has the patch's own contrast; every intensity
    augmentation after this transform then acts on fill and anatomy alike. The
    within-region variation is spatially correlated noise with the region's standard
    deviation: SynthSeg's iid draws read as grain next to the smooth scan, a cue that alone
    would separate fill from anatomy.
    """
    n, c = image.shape[:2]
    # A comparison rather than `F.one_hot`, which reads the label range back to the host on CUDA.
    member = (atlas.flatten(1).unsqueeze(2) == torch.arange(k_max, device=atlas.device)).float()  # [N, V, K], -1 matches none
    flat = image.flatten(2)  # [N, C, V]
    count = member.sum(1).clamp_min(eps).unsqueeze(1)  # [N, 1, K]
    mean = (flat @ member) / count
    std = (((flat * flat) @ member) / count - mean * mean).clamp_min(0.0).sqrt()
    index = layout.flatten(1).clamp_min(0).unsqueeze(1).expand(n, c, -1)  # -1 lies outside the fill
    texture = gaussian_blur_3d(torch.randn_like(image), torch.full((3,), _TEXTURE_SIGMA, device=image.device)).flatten(2)
    texture = texture / texture.std(dim=2, keepdim=True).clamp_min(eps)
    drawn = mean.gather(2, index) + std.gather(2, index) * texture
    sigma = torch.full((3,), _RENDER_SIGMA, device=image.device)
    return gaussian_blur_3d(drawn.view_as(image), sigma, blur_range=_RENDER_BLUR_RANGE)


class FOVExtensionGenerator3D(RandomGeneratorBase):
    """Per row: the shift as a fraction of the patch along `(d, h, w)`, and the EM region count."""

    def __init__(self, shift: tuple[float, float, float], regions: tuple[int, int]) -> None:
        super().__init__()
        self.shift = shift
        self.regions = regions

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.shift_samplers = [
            UniformDistribution(
                torch.tensor(-s, device=device, dtype=dtype), torch.tensor(s, device=device, dtype=dtype), validate_args=False
            )
            for s in self.shift
        ]
        lo, hi = self.regions
        self.regions_sampler = UniformDistribution(
            torch.tensor(float(lo), device=device, dtype=dtype),
            torch.tensor(float(hi + 1), device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        n = batch_shape[0]
        shift = torch.stack([_adapted_rsampling((n,), s, same_on_batch) for s in self.shift_samplers], dim=1)
        regions = _adapted_rsampling((n,), self.regions_sampler, same_on_batch).floor().clamp(max=float(self.regions[1]))
        return {"shift": shift, "regions": regions}


@register(
    aug_id=AugId.FOV_EXTENSION,
    backend=Backend.GPU,
    group=AugType.GEO,
    smoke_kwargs={"shift": 0.5, "regions": (2, 4), "layers": 2},
)
class RandomFOVExtensionGPU(RigidAffineAugmentationBase3D):
    """Shift the scan inside the patch and fill everything outside it with EM-synthesised background.

    Per selected sample:

    1. Image and segmentation move by the same whole-voxel shift, up to `shift` of the patch
       per axis. At `shift=1.0` the spine can leave the window entirely. With
       `spare_label_faces` the shift never vacates a window face the labels come within
       `margin` voxels of: a structure that runs out of the window may leave through that
       face, as at any window edge, but never gets fill against it, which would be an end
       the anatomy does not have. One that runs through both faces, as the spine does along
       its length, is not shifted along that axis.
    2. The *fill region* is every voxel that is not scan: the padding (every channel exactly
       `pad_value`, which is what nnU-Net pads with) plus whatever the shift vacated.
    3. The patch's own background as it arrives, before the shift -- label 0, more than
       `margin` voxels from any label, not on the padding's interpolated edge -- is split
       into a random 3 to 10 (`regions`) intensity regions by EM, as SynthSeg's label
       completion splits its background. Before, because after a large shift only a sliver
       of the scan is left in the window, and a fill built from that sliver is one region.
    4. The region map is post-processed: a 3x3x3 majority filter removes the EM speckle, and
       a random diffeomorphic field (SynthSeg's) deforms it.
    5. `layers` randomly rotated, scaled, mirrored and translated copies of it are composited
       into the fill region; any gap left takes a neighbour's region.
    6. Each fill voxel is drawn from its region's Gaussian, measured on the patch before the
       shift, with spatially correlated texture, and blurred at SynthSeg's atlas-resolution
       sigma. Its label is 0.

    Everything outside the fill region is the shifted input, bit for bit. A sample with no
    usable background still shifts and keeps the fill at `pad_value`, so image and mask agree.
    Without a segmentation (an image-only pipeline) the background is every non-padding voxel
    and no face is spared.

    **Row selection is always per sample**, whatever the pipeline's `same_on_batch` says.
    Under the paper config the trainer forces `same_on_batch=True` onto every child, which
    would make `p` one coin per batch and could empty both samples of a step of spine.

    Place it before every intensity transform, which `PIPELINE_ORDER` does, so the fill is
    augmented like real tissue. A label-to-image generator (SynthSeg, PALETTE synthesis)
    later in the pipeline regenerates the fill from label 0 and overwrites it.

    Args:
        shift: maximum shift as a fraction of the patch, one value or one per `(d, h, w)`
            axis, drawn uniformly in `[-shift, shift]`. 0 fills the existing padding only.
        regions: `(min, max)` number of EM regions, drawn per sample.
        margin: voxels between any label and the nearest background voxel used as a source,
            and how close to a window face a label counts as touching it.
        layers: warped copies of the region map composited into the fill region.
        pad_value: the intensity that marks padding.
        spare_label_faces: never vacate a window face the labels touch (step 1).
    """

    def __init__(
        self,
        shift: float | Sequence[float] = 0.5,
        regions: Sequence[int] = (3, 10),
        margin: int = 4,
        layers: int = 6,
        pad_value: float = 0.0,
        spare_label_faces: bool = True,
        p: float = 0.5,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=False, keepdim=keepdim)
        shifts = (float(shift),) * 3 if isinstance(shift, (int, float)) else tuple(float(s) for s in shift)
        if len(shifts) != 3 or not all(0.0 <= s <= 1.0 for s in shifts):
            raise ValueError(f"shift must be one value or three, each in [0, 1]. Got {shift!r}.")
        if len(regions) != 2 or not 1 <= int(regions[0]) <= int(regions[1]):
            raise ValueError(f"regions must be (min, max) with 1 <= min <= max. Got {regions!r}.")
        if int(margin) < 0 or int(layers) < 1:
            raise ValueError(f"margin must be >= 0 and layers >= 1. Got margin={margin!r}, layers={layers!r}.")
        if not isinstance(spare_label_faces, bool):
            raise TypeError(f"spare_label_faces must be true or false. Got {spare_label_faces!r}.")
        self.regions = (int(regions[0]), int(regions[1]))
        self.margin = int(margin)
        self.layers = int(layers)
        self.pad_value = float(pad_value)
        self.spare_label_faces = spare_label_faces
        self._param_generator = FOVExtensionGenerator3D(shifts, self.regions)

    @property  # type: ignore[override]
    def same_on_batch(self) -> bool:
        """Always False: see the class docstring. The setter ignores what a container writes."""
        return False

    @same_on_batch.setter
    def same_on_batch(self, value: bool) -> None:
        pass

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def _offsets(self, params: dict[str, Tensor], spatial: Sequence[int], device: torch.device | None = None) -> Tensor:
        """`[N, 3]` long voxel offsets on `device`, with the spared faces applied.

        Both hooks call this, and both read the segmentation from `params` rather than the
        mask hook's own input, so the image and the mask cannot be given different offsets.
        A spared face replaces the draw instead of redrawing it: the random stream is the
        same either way.
        """
        shift = params["shift"].detach()
        device = shift.device if device is None else device
        size = torch.tensor(list(spatial), dtype=torch.float32, device=device)
        offsets = (shift.to(device=device, dtype=torch.float32) * size).round().long()
        seg = segmentation_from(params)
        if self.spare_label_faces and seg is not None:
            if seg.shape[0] != offsets.shape[0]:
                raise ValueError(f"params['seg'] has {seg.shape[0]} rows for {offsets.shape[0]} shifts")
            touched = _touched_faces(_foreground(seg.to(device), dim=1).unsqueeze(1), self.margin)
            # A positive offset vacates the low face, a negative one the high face.
            offsets = torch.where(touched[..., 0], offsets.clamp(max=0), offsets)
            offsets = torch.where(touched[..., 1], offsets.clamp(min=0), offsets)
        return offsets

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # fp32 throughout: under the trainer's autocast `warp_volume`'s bmm and the EM
        # matmuls would run in half precision.
        with torch.autocast(device_type=input.device.type, enabled=False):
            image = input.float()
            n = image.shape[0]
            d, h, w = image.shape[2:]
            offsets = self._offsets(params, (d, h, w), image.device)
            padding = (image == self.pad_value).all(dim=1, keepdim=True)
            seg = segmentation_from(params)
            labelled = torch.zeros_like(padding) if seg is None else _foreground(seg, dim=1).unsqueeze(1)

            # The region map and its statistics come from the patch before the shift (step 3).
            sources = ~(_dilate(labelled, self.margin) | _dilate(padding, _SEAM))
            k_max = self.regions[1]
            n_regions = params["regions"].to(image.device).long()
            em = _em_regions(image[:, 0].flatten(1), sources.flatten(1).float(), n_regions, k_max).view_as(padding)

            # Majority filter over the 3x3x3 neighbourhood, among source voxels only.
            votes = (em == torch.arange(k_max, device=em.device).view(1, -1, 1, 1, 1)).float() * sources
            atlas = F.avg_pool3d(votes, 3, stride=1, padding=1).argmax(dim=1, keepdim=True)
            atlas = torch.where(sources, atlas, -1)

            displacement = random_svf_field(n, (d, h, w), image.device)
            deformed = warp_volume((atlas + 1).float(), displacement=displacement, interp="nearest").round().long() - 1

            shifted = _shift(image, offsets, self.pad_value)
            fill = _shift(padding, offsets, True)
            # The scan keeps its own regions, so what the layers miss continues them.
            seed = torch.where(fill, -1, _shift(atlas, offsets, -1))
            layout = _extend_regions(deformed, seed, fill, self.layers)

            rendered = _render(image, atlas, layout, k_max)
            has_sources = sources.flatten(1).any(dim=1).view(n, 1, 1, 1, 1)
            rendered = torch.where(has_sources, rendered, self.pad_value)
            out = torch.where(fill, rendered, shifted)
        return out.to(input.dtype)

    def apply_non_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return input

    def apply_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return _shift(input, self._offsets(params, input.shape[2:], input.device), 0)
