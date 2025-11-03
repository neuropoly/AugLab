import argparse
import importlib.resources
import shutil

import nnunetv2

import auglab.trainers as trainers

def parse_args():
    parser = argparse.ArgumentParser(
        description="This script copies an auglab nnUNetTrainer inside the nnunet folder."
    )
    parser.add_argument(
        "-t", "--trainer",
        choices=["nnUNetTrainerDAExt", "nnUNetTrainerTest"],
        type=str,
        required=True,
        help="nnUNetTrainer to be copied. Choices are: nnUNetTrainerDAExt and nnUNetTrainerTest",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Find trainer path
    trainers_path = importlib.resources.files(trainers)
    if args.trainer == "nnUNetTrainerDAExt":
        source_trainer = trainers_path / "nnUNetTrainerDAExt.py"
    elif args.trainer == "nnUNetTrainerTest":
        source_trainer = trainers_path / "nnUNetTrainerTest.py"
    else:
        raise ValueError(f"Trainer {args.trainer} not recognized.")

    # Find nnUNet path
    nnunetv2_path = importlib.resources.files(nnunetv2)
    nnunet_trainers_path = nnunetv2_path / "training" / "nnUNetTrainer"

    # Copy trainer
    output_path = nnunet_trainers_path / source_trainer.name
    if not output_path.exists():
        shutil.copy(source_trainer, output_path)
    
if __name__ == "__main__":
    main()
