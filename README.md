# AugLab
This repository investigates the influence of different data augmentation strategies on MRI training performance.

## What is available ?

This repository contains:
- A nnUNet [trainer](https://github.com/neuropoly/AugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/auglab/trainers/nnUNetTrainerDAExt.py) with extensive data augmentations
- A basic Monai segmentation [script](https://github.com/neuropoly/AugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/scripts/train_monai.py) incorporating data augmentations
- A [script](https://github.com/neuropoly/AugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/scripts/generate_augmentations.py) generating augmentations from input images and segmentations

## How to use ?

1. Open a `bash` terminal in the directory where you want to work.

2. Create and activate a virtual environment using python >=3.10 (highly recommended):
   - venv
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - conda env
   ```
   conda create -n myenv python=3.10
   conda activate myenv
   ```

3. Install this repository using one of the following options:
   - Git clone (for developpers)
   > **Note:** If you pull a new version from GitHub, make sure to rerun this command with the flag `--upgrade`
   ```bash
   git clone git@github.com:neuropoly/AugLab.git
   cd AugLab
   python3 -m pip install -e .
   ```

4. Install PyTorch following the instructions on their [website](https://pytorch.org/). Be sure to add the `--upgrade` flag to your installation command to replace any existing PyTorch installation.
   Example:
```bash
python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

