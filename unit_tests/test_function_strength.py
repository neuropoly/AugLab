"""Strength ranges for the five elementwise TAs and the unsharp mask.

Each elementwise TA min-max scales a sample to `[0, 1]`, applies a fixed function and,
with `retain_stats`, restores mean and std. Only the function's curvature on `[0, 1]`
survives that, and for log1p, sqrt, sin, exp and the sigmoid it is small: the share of
the output variance that a linear map of the input cannot explain is 0.8%, 4.0%, 0.5%,
1.6% and 0.1% on a uniform input. Each now takes an optional range that draws a
stronger member of its family. The unsharp mask truncated its blur to a 3-wide kernel
and drew sigma and amount from zero up; it now takes lower bounds and a kernel size.

The defaults must stay the old transforms exactly, including drawing no extra random
numbers, so every shipped config keeps its seeded output.
"""

from __future__ import annotations

import math

import torch

from smauglab.registry import Backend, accepted_params, get
from smauglab.transforms.gpu.contrast import (
    RandomExpGPU,
    RandomLog1pGPU,
    RandomSigmoidGPU,
    RandomSinGPU,
    RandomSqrtGPU,
    RandomUnsharpMaskGPU,
    _RandomFunctionBaseGPU,
    _restore_stats,
)
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (3, 1, 16, 16, 16)

#: leaf -> (fixed function, range parameter, a strong range)
LEAVES = {
    RandomLog1pGPU: (lambda x: torch.log(1 + x), "scale_range", [50.0, 100.0]),
    RandomSqrtGPU: (torch.sqrt, "exponent_range", [0.2, 0.25]),
    RandomSinGPU: (torch.sin, "frequency_range", [1.0, 1.5]),
    RandomExpGPU: (torch.exp, "rate_range", [4.0, 5.0]),
    RandomSigmoidGPU: (lambda x: 1 / (1 + torch.exp(-x)), "slope_range", [15.0, 20.0]),
}


def image() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.rand(*SHAPE) * 3.0 - 1.0


def nonlinear_share(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """1 - r^2 between output and input, per sample."""
    o = out.reshape(out.shape[0], -1)
    i = inp.reshape(inp.shape[0], -1)
    o = o - o.mean(1, keepdim=True)
    i = i - i.mean(1, keepdim=True)
    r = (o * i).sum(1) / (o.norm(dim=1) * i.norm(dim=1))
    return 1 - r**2


class TestDefaultsAreTheOldFunctions(SmaugLabTestCase):
    def test_default_output_is_the_fixed_function_with_stats_restored(self):
        x = image()
        flat = x[:, 0]
        lo = flat.amin(dim=(1, 2, 3), keepdim=True)
        hi = flat.amax(dim=(1, 2, 3), keepdim=True)
        normalised = (flat - lo) / (hi - lo + 0.00001)
        stats = (flat.mean(dim=(1, 2, 3)), flat.std(dim=(1, 2, 3)))
        for cls, (fixed, _, _) in LEAVES.items():
            with self.subTest(transform=cls.__name__):
                out = cls(retain_stats=True, p=1.0)(x.clone())
                expected = _restore_stats(fixed(normalised), stats)
                self.assertTrue(torch.equal(out[:, 0], expected))

    def test_defaults_draw_what_the_generic_base_draws(self):
        """Same RNG stream as `_RandomFunctionBaseGPU(func=...)`, the unchanged generic path."""
        x = image()
        for cls, (fixed, _, _) in LEAVES.items():
            with self.subTest(transform=cls.__name__):
                torch.manual_seed(0)
                leaf_out = cls(retain_stats=True, p=0.5)(x.clone())
                after_leaf = torch.rand(4)
                torch.manual_seed(0)
                base_out = _RandomFunctionBaseGPU(func=fixed, retain_stats=True, p=0.5)(x.clone())
                after_base = torch.rand(4)
                self.assertTrue(torch.equal(leaf_out, base_out))
                self.assertTrue(torch.equal(after_leaf, after_base))


class TestRangesStrengthenTheMap(SmaugLabTestCase):
    def test_a_strong_range_is_more_nonlinear_than_the_fixed_function(self):
        x = image()
        for cls, (_, key, strong) in LEAVES.items():
            with self.subTest(transform=cls.__name__):
                torch.manual_seed(0)
                fixed = nonlinear_share(cls(retain_stats=True, p=1.0)(x.clone()), x)
                torch.manual_seed(0)
                ranged = nonlinear_share(cls(retain_stats=True, p=1.0, **{key: strong})(x.clone()), x)
                self.assertTrue(bool((ranged > fixed).all()), f"{cls.__name__}: {fixed} -> {ranged}")
                self.assertTrue(bool(torch.isfinite(ranged).all()))

    def test_draws_are_per_sample_unless_same_on_batch(self):
        one = image()[:1]
        x = one.repeat(SHAPE[0], 1, 1, 1, 1)  # three copies of one sample
        for cls, (_, key, strong) in LEAVES.items():
            for same_on_batch in (False, True):
                with self.subTest(transform=cls.__name__, same_on_batch=same_on_batch):
                    torch.manual_seed(0)
                    out = cls(p=1.0, same_on_batch=same_on_batch, **{key: strong})(x.clone())
                    identical = torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])
                    self.assertEqual(identical, same_on_batch)

    def test_the_sigmoid_centre_is_the_drawn_quantile(self):
        x = image()
        sigmoid = RandomSigmoidGPU(p=1.0, slope_range=[1000.0, 1000.0], center_quantile_range=[0.3, 0.3])
        out = sigmoid(x.clone())
        above = (out[:, 0] > 0.5).reshape(SHAPE[0], -1).float().mean(1)
        torch.testing.assert_close(above, torch.full((SHAPE[0],), 0.7), atol=0.01, rtol=0)

    def test_sin_is_non_monotonic_above_half_a_cycle(self):
        ramp = torch.linspace(0, 1, 4096).view(1, 1, 16, 16, 16)
        out = RandomSinGPU(p=1.0, frequency_range=[1.5, 1.5])(ramp.clone()).flatten()
        steps = out[1:] - out[:-1]
        self.assertTrue(bool((steps > 0).any() and (steps < 0).any()))

    def test_bad_ranges_are_refused(self):
        with self.assertRaises(ValueError):
            RandomExpGPU(rate_range=[5.0, 1.0])
        with self.assertRaises(ValueError):
            RandomSqrtGPU(exponent_range=[-0.5, 0.5])
        with self.assertRaises(ValueError):
            RandomSigmoidGPU(slope_range=[4.0, 20.0], center_quantile_range=[0.5, 1.5])


class TestUnsharpMaskKnobs(SmaugLabTestCase):
    def test_default_kernel_is_three_wide(self):
        self.assertEqual(RandomUnsharpMaskGPU().get_kernel(torch.device("cpu")).shape, (3, 3, 3))

    def test_none_sizes_the_kernel_for_the_largest_sigma(self):
        transform = RandomUnsharpMaskGPU(sigma=1.5, kernel_size=None)
        k = 2 * math.ceil(3 * 1.5) + 1
        self.assertEqual(transform.get_kernel(torch.device("cpu")).shape, (k, k, k))

    def test_sigma_is_drawn_from_the_lower_bound_up(self):
        transform = RandomUnsharpMaskGPU(sigma=1.5, sigma_min=1.5, kernel_size=None)
        # sigma_min == sigma pins the draw, so the kernel is the fixed Gaussian.
        from smauglab.transforms.kernels import gaussian_kernel3d

        expected = gaussian_kernel3d(11, torch.tensor([1.5, 1.5, 1.5]), torch.float32, torch.device("cpu"))
        torch.testing.assert_close(transform.get_kernel(torch.device("cpu")), expected)

    def test_a_wider_blur_sharpens_more(self):
        x = image()
        torch.manual_seed(0)
        weak = RandomUnsharpMaskGPU(p=1.0, same_on_batch=True)(x.clone())
        torch.manual_seed(0)
        strong = RandomUnsharpMaskGPU(
            p=1.0, same_on_batch=True, sigma=1.5, sigma_min=0.5, unsharp_amount=2.0, amount_min=0.5, kernel_size=None
        )(x.clone())
        self.assertGreater(float((strong - x).abs().mean()), float((weak - x).abs().mean()))

    def test_bad_bounds_are_refused(self):
        with self.assertRaises(ValueError):
            RandomUnsharpMaskGPU(sigma=1.0, sigma_min=2.0)
        with self.assertRaises(ValueError):
            RandomUnsharpMaskGPU(unsharp_amount=1.0, amount_min=2.0)
        with self.assertRaises(ValueError):
            RandomUnsharpMaskGPU(kernel_size=4)


class TestConfigSurface(SmaugLabTestCase):
    def test_the_new_parameters_are_accepted(self):
        expected = {
            "RandomLog1pGPU": {"scale_range"},
            "RandomSqrtGPU": {"exponent_range"},
            "RandomSinGPU": {"frequency_range"},
            "RandomExpGPU": {"rate_range"},
            "RandomSigmoidGPU": {"slope_range", "center_quantile_range"},
            "RandomUnsharpMaskGPU": {"sigma_min", "amount_min", "kernel_size"},
        }
        for name, keys in expected.items():
            with self.subTest(transform=name):
                self.assertLessEqual(keys, set(accepted_params(get(name, Backend.GPU))))
