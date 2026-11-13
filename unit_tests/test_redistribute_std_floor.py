"""`RandomRedistributeSegGPU(min_redist_std=...)` bounds how steep the remap gets.

The remap adds a signed Gaussian bump per label, centred on that label's mean intensity,
and rescales the sum to a peak-to-peak of 2 in the `[0, 1]` min-max units it works in.
Each bump's width is floored at `min_redist_std`. A bump of width s, rescaled that way,
has a slope of about 1.2 / s, so the historical floor of 0.01 lets a thin band of
intensities swing by more than the whole input range; voxel noise inside the band then
comes out as salt-and-pepper. The floor binds whenever a label's mean is close to its
dilation ring's and the random part of the width is drawn small.
"""

from __future__ import annotations

import torch

from smauglab.registry import Backend, accepted_params, get
from smauglab.transforms.gpu.fromSeg import RandomRedistributeSegGPU
from unit_tests.helpers import SmaugLabTestCase

N = 16


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """An intensity ramp and a voxel-random label map.

    Random labels make every region's dilation ring look like the region itself, so the
    data-driven part of the width is about zero and the floor is what sets it.
    """
    torch.manual_seed(0)
    image = torch.linspace(0, 1, N**3).view(1, 1, N, N, N)
    labels = torch.randint(0, 3, (1, 1, N, N, N)).float()
    return image, labels


def max_slope(floor: float, seed: int) -> float:
    image, labels = inputs()
    torch.manual_seed(seed)
    transform = RandomRedistributeSegGPU(p=1.0, in_seg=0.0, retain_stats=False, std_noise_range=(0.0, 0.0), min_redist_std=floor)
    out = transform.apply_transform(image.clone(), {"seg": labels}, transform.flags).flatten()
    x = image.flatten()
    return float(((out[1:] - out[:-1]) / (x[1:] - x[:-1])).abs().max())


class TestRedistributeStdFloor(SmaugLabTestCase):
    def test_the_slope_follows_the_floor(self):
        for floor in (0.01, 0.05, 0.1):
            with self.subTest(floor=floor):
                slopes = [max_slope(floor, seed) for seed in range(10)]
                # 1 from the identity plus about 1.2 / floor from the steepest bump.
                self.assertLess(max(slopes), 1.5 * (1 + 1.2 / floor))
                self.assertGreater(max(slopes), 0.5 * (1 + 1.2 / floor))

    def test_the_default_is_the_historical_floor(self):
        image, labels = inputs()
        outs = []
        for kwargs in ({}, {"min_redist_std": 0.01}):
            torch.manual_seed(1)
            transform = RandomRedistributeSegGPU(p=1.0, **kwargs)
            outs.append(transform.apply_transform(image.clone(), {"seg": labels}, transform.flags))
        self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_a_non_positive_floor_is_refused(self):
        with self.assertRaises(ValueError):
            RandomRedistributeSegGPU(min_redist_std=0.0)

    def test_the_config_surface_accepts_it(self):
        self.assertIn("min_redist_std", accepted_params(get("RandomRedistributeSegGPU", Backend.GPU)))
