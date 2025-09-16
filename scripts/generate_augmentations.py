import argparse, textwrap
import json
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import numpy as np
import warnings

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    NormalizeIntensityd,
)

from auglab.utils.utils import fetch_image_config
from auglab.transforms.transforms import get_train_transforms
from auglab.utils.image import Image, resample_nib

warnings.filterwarnings("ignore")

rs = np.random.RandomState()

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
            It apply transformation on the image and the segmentation to make augmented image. Useful if augmentations cannot be performed on the fly during training. 
            Based on https://github.com/neuropoly/totalspineseg/blob/b4da40840ad618498be3ec02564d4c5f5fa5c8aa/totalspineseg/utils/augment.py
        '''.split()),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--data', '-d', type=Path, required=True,
        help='Data config JSON file containing the IMAGE and LABEL paths of the files used. Only TRAINING will be augmented. See example in auglab.configs.data (required).'
    )
    parser.add_argument(
        '--ofolder', '-o', type=Path, required=True,
        help='The folder where output augmented images will be saved with _a1, _a2 etc. suffixes (required).'
    )
    parser.add_argument(
        '--transforms', '-t', type=Path, required=True,
        help='Transforms config JSON file containing the parameters of the transformations. See example in auglab.configs (required).'
    )
    parser.add_argument(
        '--augmentations-per-image', '-n', type=int, default=5,
        help='Number of augmentation images to generate. Default is 5.'
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='If provided, overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    data_json_path = args.data
    transforms_json_path = args.transforms
    ofolder = args.ofolder
    augmentations_per_image = args.augmentations_per_image
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            data_json_path = {data_json_path}
            transforms_json_path = {transforms_json_path}
            ofolder = {ofolder}
            augmentations_per_image = {augmentations_per_image}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    augment_mp(
        data_json_path=data_json_path,
        transforms_json_path=transforms_json_path,
        ofolder=ofolder,
        augmentations_per_image=augmentations_per_image,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def augment_mp(
        data_json_path,
        transforms_json_path,
        ofolder,
        augmentations_per_image=5,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    # Load data config
    with open(data_json_path, "r") as f:
        data_config = json.load(f)
    
    data_list, _ = fetch_image_config(
        config_data=data_config,
        split='TRAINING',
    )

    # Init transforms
    if not transforms_json_path.is_file():
        print(f'Error: {transforms_json_path}, Transforms config file not found')
        return
    
    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            # Insert AugLab transforms here
        ] + get_train_transforms(json_path=str(transforms_json_path))
    )

    process_map(
        partial(
            augment,
            augmentations_per_image=augmentations_per_image,
            ofolder=ofolder,
            train_transforms=train_transforms,
            overwrite=overwrite,
        ),
        data_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def augment(
        data_dict,
        augmentations_per_image,
        train_transforms,
        ofolder,
        overwrite=False,
    ):
    '''
    Augmentation function.
    '''
    img_path = Path(data_dict['image'])
    seg_path = Path(data_dict['label'])

    # Load images
    img = Image(str(img_path)).change_orientation('RPI') # RPI- == LAS+
    seg = Image(str(seg_path)).change_orientation('RPI')

    # Resample to 1mm isotropic
    pr = 1.0
    img = resample_nib(img, new_size=[pr, pr, pr], new_size_type='mm', interpolation='spline', verbose=False)
    seg = resample_nib(seg, new_size=[pr, pr, pr], new_size_type='mm', interpolation='nn', verbose=False)

    # Create augmentations
    for i in range(augmentations_per_image):
        # Create output path
        output_image_path = Path(ofolder) / f"{img_path.name}_a{i+1}"
        output_seg_path = Path(ofolder) / f"{seg_path.name}_a{i+1}"

        # Generate augmentation
        if not overwrite and (output_image_path.exists() or output_seg_path.exists()):
            continue
        
        # Transform data
        tensor_dict = train_transforms({'image': img.data, 'label': seg.data})


if __name__ == '__main__':
    #main()
    transforms_json_path = Path('auglab/configs/transform_params.json')
    augment(
        data_dict={
            'image': '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/datasets/Task01_BrainTumour/imagesTr/brats_001.nii.gz',
            'label': '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/datasets/Task01_BrainTumour/labelsTr/brats_001.nii.gz',
        },
        augmentations_per_image=2,
        train_transforms=Compose(
            [
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
                # Insert AugLab transforms here
            ] + get_train_transforms(json_path=str(transforms_json_path))
        ),
        ofolder='/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-auglab/augmented',
        overwrite=True,
    )