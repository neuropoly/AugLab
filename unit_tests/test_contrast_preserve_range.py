"""`RandomContrastGPU(preserve_range=True)` clips to each sample's original range, as nnU-Net does.

Without the clip the transform is `factor * x + (1 - factor) * mean`, a global affine map,
which a conv -> InstanceNorm network cancels except at the patch border. nnU-Net's
`ContrastTransform(preserve_range=True)` clamps the result back to the input's [min, max],
so a factor above 1 saturates both tails. The default stays unclipped and draws the same
random numbers, so every shipped config keeps its seeded output.
"""

from __future__ import annotations

import torch
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform

from smauglab.registry import Backend, accepted_params, get
from smauglab.transforms.gpu.contrast import RandomContrastGPU
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (3, 1, 16, 16, 16)
SPATIAL = (1, 2, 3, 4)


def image() -> torch.Tensor:
    """Three samples with different ranges, so a batch-wide clip would show."""
    torch.manual_seed(7)
    scale = torch.tensor([0.5, 1.0, 3.0]).view(-1, 1, 1, 1, 1)
    offset = torch.tensor([-1.0, 0.0, 2.0]).view(-1, 1, 1, 1, 1)
    return torch.randn(*SHAPE) * scale + offset


def contrast(x: torch.Tensor, factor: float, **kwargs) -> torch.Tensor:
    return RandomContrastGPU(contrast_range=(factor, factor), p=1.0, **kwargs)(x.clone())


def affine(x: torch.Tensor, factor: float) -> torch.Tensor:
    mean = x.mean(dim=SPATIAL, keepdim=True)
    return (x - mean) * factor + mean


class TestDefaultIsUnclipped(SmaugLabTestCase):
    def test_default_is_the_affine_map_and_leaves_the_range(self):
        x = image()
        out = contrast(x, 1.5)
        torch.testing.assert_close(out, affine(x, 1.5))
        self.assertTrue(bool((out.amax(dim=SPATIAL) > x.amax(dim=SPATIAL)).all()))
        self.assertTrue(bool((out.amin(dim=SPATIAL) < x.amin(dim=SPATIAL)).all()))

    def test_the_clip_draws_no_random_numbers(self):
        x = image()
        draws = []
        for preserve_range in (False, True):
            torch.manual_seed(0)
            RandomContrastGPU(p=0.5, preserve_range=preserve_range)(x.clone())
            draws.append(torch.rand(4))
        self.assertTrue(torch.equal(draws[0], draws[1]))


class TestPreserveRange(SmaugLabTestCase):
    def test_matches_nnunets_contrast_transform(self):
        x = image()
        for factor in (0.75, 1.5):
            with self.subTest(factor=factor):
                nnunet = ContrastTransform((factor, factor), preserve_range=True, synchronize_channels=True)
                expected = torch.stack([nnunet(image=sample.clone())["image"] for sample in x])
                torch.testing.assert_close(contrast(x, factor, preserve_range=True), expected)

    def test_each_sample_is_clipped_to_its_own_range(self):
        x = image()
        lo, hi = x.amin(dim=SPATIAL, keepdim=True), x.amax(dim=SPATIAL, keepdim=True)
        for same_on_batch in (False, True):
            with self.subTest(same_on_batch=same_on_batch):
                out = contrast(x, 1.5, preserve_range=True, same_on_batch=same_on_batch)
                torch.testing.assert_close(out, torch.clamp(affine(x, 1.5), lo, hi))
                clipped = (out == lo) | (out == hi)
                self.assertTrue(bool((clipped.sum(dim=SPATIAL) > 1).all()), "a factor of 1.5 should saturate more than the extremes")

    def test_a_factor_below_one_needs_no_clip(self):
        x = image()
        self.assertTrue(torch.equal(contrast(x, 0.75, preserve_range=True), contrast(x, 0.75)))

    def test_clip_comes_before_retain_stats(self):
        """Restoring the stats undoes the affine map; only the clipped tails survive it."""
        x = image()
        restored = contrast(x, 1.5, retain_stats=True)
        torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)

        out = contrast(x, 1.5, retain_stats=True, preserve_range=True)
        torch.testing.assert_close(out.mean(dim=SPATIAL), x.mean(dim=SPATIAL), atol=1e-5, rtol=0)
        torch.testing.assert_close(out.std(dim=SPATIAL), x.std(dim=SPATIAL), atol=1e-5, rtol=1e-5)
        self.assertGreater(float((out - x).abs().amax()), 0.05)

    def test_the_config_surface_accepts_it(self):
        self.assertIn("preserve_range", accepted_params(get("RandomContrastGPU", Backend.GPU)))
