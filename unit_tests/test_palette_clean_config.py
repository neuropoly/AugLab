"""`transform_params_paper-palette-clean.json`: PALETTE alone without its artificial edges.

The config is `transform_params_paper-synthseg.json` with `RandomSynthSegGPU` replaced by
`RandomPaletteGPU` -- the `Palette alone` arm, hash `4b92045c` -- plus three PALETTE keys:

* `skip_sub_parc_prob: 1.0` -- no Voronoi sub-parcellation. The split cuts straight
  polygon edges through anatomy, edges no imaging modality produces.
* `label_remap_prob: 0.0` -- no per-label affine remap, which steps the intensity exactly
  on the ground-truth boundary.
* `blur_sigmas: [0.5, 0.8]` -- a blur floor. The k-means bucketing is hard, so voxels
  next to a cluster threshold flip between two remaps and come out as salt-and-pepper.

With the first two off, the synthesis before the blur is a function of the input
intensity alone, which is what the tests below check.
"""

from __future__ import annotations

import json

import torch

from smauglab.transforms.gpu.fromSeg import RandomPaletteGPU
from smauglab.transforms.gpu.transforms import AugTransformsGPU
from unit_tests.helpers import SmaugLabTestCase, configs_dir

CONFIG = "transform_params_paper-palette-clean.json"
SHAPE = (1, 1, 16, 16, 16)
LEVELS = torch.tensor([0.2, 0.35, 0.5, 0.65, 0.8, 1.0])


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Six intensity levels scattered over two labels, so a level appears in both."""
    torch.manual_seed(5)
    image = LEVELS[torch.randint(0, len(LEVELS), SHAPE)]
    image[0, 0, 0, 0, 0] = 0.0  # a dark voxel, so the min-max range is fixed
    labels = torch.ones(*SHAPE)
    labels[..., 8:] = 2.0
    return image, labels


def spread_within_levels(out: torch.Tensor, image: torch.Tensor) -> float:
    """Largest output range among voxels that shared an input intensity."""
    return max(float(out[image == level].max() - out[image == level].min()) for level in LEVELS)


def palette(**kwargs) -> RandomPaletteGPU:
    return RandomPaletteGPU(p=1.0, blur_sigmas=(0.0,), **kwargs)


class TestPaletteCleanConfig(SmaugLabTestCase):
    def test_the_config_sets_the_three_knobs(self):
        cfg = json.loads((configs_dir() / CONFIG).read_text())
        block = cfg["GPU"]["RandomPaletteGPU"]
        self.assertEqual(block["skip_sub_parc_prob"], 1.0)
        self.assertEqual(block["label_remap_prob"], 0.0)
        self.assertGreaterEqual(min(block["blur_sigmas"]), 0.5)
        self.assertNotIn("RandomSynthSegGPU", cfg["GPU"])

    def test_the_config_builds_and_runs(self):
        pipeline = AugTransformsGPU(json_path=str(configs_dir() / CONFIG))
        out = pipeline(self.tiny_volume(), self.tiny_seg())
        image = out[0] if isinstance(out, (tuple, list)) else out
        self.assertTrue(bool(torch.isfinite(image).all()))

    def test_without_voronoi_and_label_remap_the_map_depends_on_intensity_only(self):
        image, labels = inputs()
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                out = palette(skip_sub_parc_prob=1.0, label_remap_prob=0.0).apply_transform(image.clone(), params={"seg": labels}, flags={})
                self.assertLess(spread_within_levels(out, image), 1e-5)

    def test_the_defaults_do_break_that(self):
        """The guard against a check that cannot fail: with the defaults it does."""
        image, labels = inputs()
        spreads = []
        for seed in range(5):
            torch.manual_seed(seed)
            out = palette().apply_transform(image.clone(), params={"seg": labels}, flags={})
            spreads.append(spread_within_levels(out, image))
        self.assertGreater(max(spreads), 1e-3)
