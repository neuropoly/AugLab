"""`RandomGaussianNoiseGPU(relative=True)` scales the noise with the image.

The std is drawn from `U(0, std)` in absolute intensity units, so the same config is a
different augmentation on every intensity scale: a z-scored MR patch (std ~0.9), a CT
patch under nnU-Net's CT normalisation (std ~0.17) and a `[0, 1]` SynthSeg synthesis
(std ~0.18) all get noise of the same absolute std. `relative=True` reads `std` as a
fraction of each sample's own std, which makes the transform scale-equivariant.
"""

from __future__ import annotations

import torch

from smauglab.registry import Backend, accepted_params, get
from smauglab.transforms.gpu.contrast import RandomGaussianNoiseGPU
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (4, 1, 12, 12, 12)


def noised(x: torch.Tensor, relative: bool, same_on_batch: bool, seed: int = 0) -> torch.Tensor:
    transform = RandomGaussianNoiseGPU(std=0.3, relative=relative, same_on_batch=same_on_batch, p=1.0)
    torch.manual_seed(seed)
    return transform(x.clone())


def image() -> torch.Tensor:
    torch.manual_seed(123)
    # Four samples on four different intensity scales.
    return torch.randn(*SHAPE) * torch.tensor([0.17, 0.9, 2.5, 30.0]).view(-1, 1, 1, 1, 1)


class TestRelativeGaussianNoise(SmaugLabTestCase):
    def test_the_absolute_form_is_unchanged(self):
        """Same draws in the same order, same arithmetic as before the option existed."""
        x = image()
        for same_on_batch in (False, True):
            with self.subTest(same_on_batch=same_on_batch):
                torch.manual_seed(0)
                n = 1 if same_on_batch else SHAPE[0]
                std = torch.rand(n) * 0.3
                noise = torch.randn_like(x[:, 0])
                scale = std if same_on_batch else std.view(-1, 1, 1, 1)
                expected = x[:, 0] + (noise * scale + 0.0)
                out = noised(x, relative=False, same_on_batch=same_on_batch)
                self.assertTrue(torch.equal(out[:, 0], expected))

    def test_relative_noise_is_scale_equivariant(self):
        x = image()
        for same_on_batch in (False, True):
            with self.subTest(same_on_batch=same_on_batch):
                out = noised(x, relative=True, same_on_batch=same_on_batch)
                out_scaled = noised(7.0 * x, relative=True, same_on_batch=same_on_batch)
                torch.testing.assert_close(out_scaled, 7.0 * out, rtol=1e-5, atol=1e-5)

    def test_relative_is_the_absolute_draw_times_the_sample_std(self):
        """No extra random draw: the noise field is the absolute one, rescaled per sample.

        In float64, because the noise is read back as `out - x`, which in float32 loses
        about ulp(x) -- 2e-6 at the std-30 sample, then multiplied by 30.
        """
        x = image().double()
        sample_std = x.reshape(SHAPE[0], -1).std(dim=1).view(-1, 1, 1, 1, 1)
        for same_on_batch in (False, True):
            with self.subTest(same_on_batch=same_on_batch):
                absolute = noised(x, relative=False, same_on_batch=same_on_batch) - x
                relative = noised(x, relative=True, same_on_batch=same_on_batch) - x
                torch.testing.assert_close(relative, absolute * sample_std, rtol=1e-4, atol=1e-5)

    def test_the_config_surface_accepts_it(self):
        self.assertIn("relative", accepted_params(get("RandomGaussianNoiseGPU", Backend.GPU)))
