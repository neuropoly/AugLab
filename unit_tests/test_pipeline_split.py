"""`pipeline.split`: mutually exclusive paths through the GPU pipeline.

The question it exists for: "SynthSeg on half the batches, our transfer augmentations
on the other half, the general enhancements on all of them". Stacking SynthSeg at
p=0.5 onto the paper pipeline is not that -- the transfer augmentations then also run on
the synthetic image -- and no other configuration surface could say "not both".

What is pinned here: the config refuses every malformed split at load time; the paths
never co-occur; the shared stages run on every batch; the path frequencies follow `p`;
a skipped stage touches neither image nor mask; and the trainer, the one place a split
changes what a model is trained on, passes it through.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import ClassVar
from unittest import mock

import torch

from smauglab.config import PipelineMode, SmaugConfig, SplitSpec
from smauglab.registry import Backend, InvalidConfigError
from smauglab.transforms.build import build_gpu_pipeline
from smauglab.transforms.gpu.base import AugmentationSequentialCustom, record_applications

GPU = {
    "RandomFlipTransformGPU": {"flip_axis": [0], "p": 1.0},
    "RandomInverseGPU": {"p": 1.0},
    "RandomGammaGPU": {"gamma_range": [0.7, 1.5], "p": 1.0},
    "RandomGaussianNoiseGPU": {"mean": 0.0, "std": 0.1, "p": 1.0},
}
SHAPE = torch.Size((2, 1, 8, 8, 8))


def config(split: dict | None, **pipeline) -> SmaugConfig:
    payload: dict = {"GPU": json.loads(json.dumps(GPU))}
    if split is not None:
        pipeline["split"] = split
    if pipeline:
        payload["pipeline"] = pipeline
    return SmaugConfig(payload)


def pipeline(cfg: SmaugConfig) -> AugmentationSequentialCustom:
    return AugmentationSequentialCustom(
        *build_gpu_pipeline(cfg.section(Backend.GPU), mode=cfg.pipeline_mode()),
        data_keys=["input", "mask"],
        same_on_batch=True,
        split=cfg.pipeline_split(),
    )


def stages(pipe: AugmentationSequentialCustom) -> set[str]:
    """The class names of the stages one parameter draw selected."""
    return {param.name.rsplit("_", 1)[0] for param in pipe.forward_parameters(SHAPE)}


class TestSplitConfig(unittest.TestCase):
    SPLIT: ClassVar[dict] = {"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [0.5, 0.5]}

    def assertRejected(self, split: dict | object, fragment: str, **pipeline) -> None:
        with self.assertRaises(InvalidConfigError) as caught:
            config(split, **pipeline)  # type: ignore[arg-type]
        self.assertIn(fragment, str(caught.exception))

    def test_absent_means_no_split(self):
        self.assertIsNone(config(None).pipeline_split())

    def test_a_valid_split_parses(self):
        self.assertEqual(
            config(self.SPLIT).pipeline_split(),
            SplitSpec(paths=(("RandomInverseGPU",), ("RandomGammaGPU",)), p=(0.5, 0.5)),
        )

    def test_an_empty_path_is_allowed(self):
        spec = config({"paths": [["RandomInverseGPU"], []], "p": [0.3, 0.7]}).pipeline_split()
        self.assertEqual(spec, SplitSpec(paths=(("RandomInverseGPU",), ()), p=(0.3, 0.7)))

    def test_a_name_missing_from_the_gpu_section_is_rejected_with_a_hint(self):
        self.assertRejected({"paths": [["RandomInversGPU"], ["RandomGammaGPU"]], "p": [0.5, 0.5]}, "Did you mean: RandomInverseGPU")

    def test_a_name_in_two_paths_is_rejected(self):
        self.assertRejected({"paths": [["RandomInverseGPU"], ["RandomInverseGPU"]], "p": [0.5, 0.5]}, "named twice")

    def test_probabilities_must_sum_to_one(self):
        self.assertRejected({"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [0.5, 0.4]}, "sum to 1")

    def test_probabilities_must_lie_in_the_unit_interval(self):
        self.assertRejected({"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [1.5, -0.5]}, "[0, 1]")

    def test_one_probability_per_path(self):
        self.assertRejected({"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [1.0]}, "2 paths but 1 probabilities")

    def test_a_single_path_is_not_a_split(self):
        self.assertRejected({"paths": [["RandomInverseGPU"]], "p": [1.0]}, "at least two")

    def test_an_unknown_key_is_rejected(self):
        self.assertRejected({**self.SPLIT, "weights": [1, 1]}, "unknown key 'weights'")

    def test_a_non_object_is_rejected(self):
        self.assertRejected(["RandomInverseGPU"], "expected an object")

    def test_a_bucketing_mode_is_rejected(self):
        self.assertRejected(self.SPLIT, "needs the sequential pipeline", mode=PipelineMode.RANDOM_ORDER.value)

    def test_per_sample_draws_are_rejected(self):
        self.assertRejected(self.SPLIT, "same_on_batch: false", same_on_batch=False)


class TestSplitContainer(unittest.TestCase):
    def test_no_split_keeps_every_stage(self):
        pipe = pipeline(config(None))
        self.assertEqual(stages(pipe), set(GPU))

    def test_paths_never_co_occur_and_shared_stages_always_run(self):
        pipe = pipeline(config({"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [0.25, 0.75]}))
        torch.manual_seed(0)
        n, taken = 4000, Counter()
        for _ in range(n):
            drawn = stages(pipe)
            self.assertTrue({"RandomFlipTransformGPU", "RandomGaussianNoiseGPU"} <= drawn, drawn)
            self.assertEqual(len(drawn & {"RandomInverseGPU", "RandomGammaGPU"}), 1, drawn)
            taken["inverse" if "RandomInverseGPU" in drawn else "gamma"] += 1
        # Binomial(4000, 0.25): sd 27.4, so +-150 is over five sd.
        self.assertLess(abs(taken["inverse"] - 0.25 * n), 150, taken)

    def test_an_empty_path_runs_neither_of_the_others(self):
        pipe = pipeline(config({"paths": [["RandomInverseGPU", "RandomGammaGPU"], []], "p": [0.5, 0.5]}))
        torch.manual_seed(1)
        seen = Counter(frozenset(stages(pipe) & {"RandomInverseGPU", "RandomGammaGPU"}) for _ in range(400))
        self.assertEqual(set(seen), {frozenset({"RandomInverseGPU", "RandomGammaGPU"}), frozenset()}, seen)

    def test_the_draw_follows_the_torch_seed(self):
        pipe = pipeline(config({"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [0.5, 0.5]}))
        runs = []
        for _ in range(2):
            torch.manual_seed(7)
            runs.append([frozenset(stages(pipe)) for _ in range(50)])
        self.assertEqual(runs[0], runs[1])

    def test_a_skipped_stage_touches_neither_image_nor_mask(self):
        pipe = pipeline(config({"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [0.5, 0.5]}))
        torch.manual_seed(2)
        x = torch.rand(SHAPE) + 0.1
        seg = (torch.rand(SHAPE) > 0.5).float()
        taken = Counter()
        for _ in range(40):
            with record_applications(pipe) as fired:
                _, mask = pipe(x.clone(), seg.clone())
            ran = {entry["transform"] for entry in fired if entry["target"] == "image"}
            self.assertEqual(len(ran & {"RandomInverseGPU", "RandomGammaGPU"}), 1, ran)
            self.assertIn("RandomGaussianNoiseGPU", ran)
            taken[frozenset(ran)] += 1
            # Flip at p=1 is the only stage that moves the mask, whichever path ran.
            self.assertTrue(torch.equal(mask, seg.flip(2)))
        self.assertEqual(len(taken), 2, taken)

    def test_a_name_that_is_not_a_stage_raises(self):
        # In a bucketing mode the members live inside a RandomChooseX and are not stages;
        # gating them would silently do nothing. The config refuses this combination, so
        # this is the direct-construction guard.
        cfg = config(None)
        built = build_gpu_pipeline(cfg.section(Backend.GPU), mode=PipelineMode.RANDOM_ORDER_TA)
        spec = SplitSpec(paths=(("RandomInverseGPU",), ("RandomGammaGPU",)), p=(0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "not a stage"):
            AugmentationSequentialCustom(*built, data_keys=["input", "mask"], same_on_batch=True, split=spec)

    def test_aug_transforms_gpu_reads_the_split(self):
        from smauglab.transforms.gpu.transforms import AugTransformsGPU

        payload = {"GPU": GPU, "pipeline": {"split": {"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [0.5, 0.5]}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transform_params_split_aug.json"
            path.write_text(json.dumps(payload))
            pipe = AugTransformsGPU(str(path))
        self.assertEqual(pipe.split_spec, SmaugConfig(payload).pipeline_split())


@unittest.skipIf(importlib.util.find_spec("nnunetv2") is None, "the trainer needs the nnunetv2 extra")
class TestTrainerPassesTheSplit(unittest.TestCase):
    """The trainer builds its own container rather than an `AugTransformsGPU`, so it has to
    pass the split itself -- `pipeline.order` was once dropped at exactly this spot."""

    def test_the_trainer_pipeline_carries_the_split(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        from smauglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

        payload = {"GPU": GPU, "pipeline": {"split": {"paths": [["RandomInverseGPU"], ["RandomGammaGPU"]], "p": [0.5, 0.5]}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transform_params_split_trainer.json"
            path.write_text(json.dumps(payload))

            def bare_init(self, *_args, **_kwargs):
                self.device = torch.device("cpu")
                self.output_folder = tmp

            with mock.patch.object(nnUNetTrainer, "__init__", bare_init), mock.patch.dict(os.environ, {"SMAUGLAB_PARAMS_JSON": str(path)}):
                trainer = nnUNetTrainerDAExtGPU({}, "3d_fullres", 0, {})

        assert trainer.transforms is not None
        self.assertEqual(trainer.transforms.split_spec, SmaugConfig(payload).pipeline_split())
        self.assertEqual(len(trainer.transforms._split_paths or ()), 2)


if __name__ == "__main__":
    unittest.main()
