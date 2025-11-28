# AugLab
This repository investigates the influence of different data augmentation strategies on MRI training performance.

## What is available ?

This repository contains:
- A nnUNet [trainer](https://github.com/neuropoly/AugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/auglab/trainers/nnUNetTrainerDAExt.py) with extensive data augmentations
- A basic Monai segmentation [script](https://github.com/neuropoly/AugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/scripts/train_monai.py) incorporating data augmentations
- A [script](https://github.com/neuropoly/AugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/scripts/generate_augmentations.py) generating augmentations from input images and segmentations

## How to install ?

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

3. Clone this repository:
   - Git clone
   ```bash
   git clone git@github.com:neuropoly/AugLab.git
   cd AugLab
   ```

4. Install AugLab using one of the following commands:
   > **Note:** If you pull a new version from GitHub, make sure to rerun this command with the flag `--upgrade`
   - nnunetv2 only usage
   ```bash
   python3 -m pip install -e .[nnunetv2]
   ```
   - full usage (with Monai and other dependencies)
   ```bash
   python3 -m pip install -e .[all]
   ```

5. Install PyTorch following the instructions on their [website](https://pytorch.org/). Be sure to add the `--upgrade` flag to your installation command to replace any existing PyTorch installation.
   Example:
```bash
python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

## Run nnunet training with AugLab trainer

To use the AugLab trainer with nnUNet, first add the trainer to your nnUNet installation by running:
```bash
auglab_add_nnunettrainer --trainer nnUNetTrainerDAExt
```

Then, when you run nnUNet training as usual, specifying the AugLab trainer, for example:
```bash
nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDAExtGPU -p nnUNetPlans
```

You can also specify your data augmentation parameters by providing a JSON file using the environment variable `AUGLAB_PARAMS_GPU_JSON`:
> **Note:** By default [auglab/configs/transform_params_gpu.json](https://github.com/neuropoly/AugLab/blob/main/auglab/configs/transform_params_gpu.json) is used if no file is specified.
```bash
AUGLAB_PARAMS_GPU_JSON=/path/to/your/params.json nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDAExtGPU -p nnUNetPlans
```

> ⚠️ **Warning** : To avoid any paths issues, please specify an absolute path to your JSON file.


## How to use my data ?

Scripts developped in this repository use JSON files to specify image and segmentation paths: see this [example](https://github.com/neuropoly/AugLab/blob/16653a84e031c40e25a72e946c2724494606b21c/auglab/configs/data/data.json).

## How do I specify my parameters ?

To track parameters used during data augmentation, JSON files are also used: see this [example](https://github.com/neuropoly/AugLab/blob/16653a84e031c40e25a72e946c2724494606b21c/auglab/configs/transform_params.json)

