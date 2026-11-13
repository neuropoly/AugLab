"""`RandomSynthSegGPU(zscore_output=True)` hands on a z-scored synthesis.

With SynthSeg's default `normalise=True` the synthesis is min-max scaled to `[0, 1]`,
while the real patches it is mixed with are z-scored. Every later transform that works
in absolute units then acts on the synthetic samples at several times its strength on
the real ones; `RandomGaussianNoiseGPU`, whose std is drawn from `U(0, std)`, buries a
`[0, 1]` image under noise in most of its firings. `zscore_output` rescales the synthesis
per sample, and does nothing else: it draws no random numbers, so the same seed gives
the same synthesis up to that rescale.
"""

from __future__ import annotations

import torch

from smauglab.registry import Backend, accepted_params, get
from smauglab.transforms.synthseg.transforms import RandomSynthSegGPU
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (3, 1, 16, 16, 16)


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros(*SHAPE)
    labels[:, :, 4:12, 4:12, 4:12] = 1
    labels[:, :, 6:10, 6:10, 6:10] = 2
    return torch.randn(*SHAPE), labels


def synthesise(zscore_output: bool, seed: int = 0) -> torch.Tensor:
    transform = RandomSynthSegGPU(p=1.0, n_channels=1, zscore_output=zscore_output)
    image, labels = inputs()
    torch.manual_seed(seed)
    return transform(image.clone(), params={"seg": labels})


def per_sample(x: torch.Tensor, reduce) -> torch.Tensor:
    return reduce(x.reshape(x.shape[0], -1), dim=1)


class TestSynthSegZscoreOutput(SmaugLabTestCase):
    def test_the_default_still_returns_the_unit_interval(self):
        out = synthesise(zscore_output=False)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_each_sample_is_zscored(self):
        out = synthesise(zscore_output=True)
        torch.testing.assert_close(per_sample(out, torch.mean), torch.zeros(SHAPE[0]), atol=1e-5, rtol=0)
        torch.testing.assert_close(per_sample(out, torch.std), torch.ones(SHAPE[0]), atol=1e-5, rtol=0)

    def test_it_is_only_a_rescale_of_the_same_synthesis(self):
        """No extra random draw: the same seed gives the same image, standardised."""
        for seed in range(3):
            with self.subTest(seed=seed):
                plain = synthesise(zscore_output=False, seed=seed)
                scored = synthesise(zscore_output=True, seed=seed)
                flat = plain.reshape(SHAPE[0], -1)
                expected = (flat - flat.mean(1, keepdim=True)) / flat.std(1, keepdim=True)
                torch.testing.assert_close(scored.reshape(SHAPE[0], -1), expected, atol=1e-5, rtol=1e-5)

    def test_the_config_surface_accepts_it(self):
        entry = get("RandomSynthSegGPU", Backend.GPU)
        self.assertIn("zscore_output", accepted_params(entry))
