![example workflow](https://github.com/numisveinsson/SeqSeg/actions/workflows/python-app.yml/badge.svg)

# SeqSeg: Automatic TracKing and Segmentation of Blood Vessels

See paper [here](https://rdcu.be/dU0wy).

Below is an example output of the algorithm on a 3D MR image of the abdominal aorta:

![](assets/mr_model_tracing_fast_shorter.gif)

## Set Up
SeqSeg relies on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for segmentation of the medical image volumes. You will need model weights to run the algorithm. After training the nnU-Net model, the weights will be saved in the `nnUNet_results` folder. Before running you must set a global environment variable to say where that folder is, for example:

```bash
export nnUNet_results="/path/to/model/weights/nnUnet/nnUNet_results"
```

Main package dependencies (see environment.yml file for all):

Machine Learning:
- nnU-Net
- Pytorch

Image and Data Processing:
- SITK
- VTK
- Numpy
- Matplotlib
- Pyyaml

and if using VMTK (not required):
- VMTK

Current workflow:
1. Create conda environment using environment_new.yml
2. Test this environment using the test script tests/test.sh
3. Install nnunet and pytorch using the instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).
4. Try the full installation according to details below


## Running

### Set weights directory
```bash
export nnUNet_results="/path/to/model/weights/nnUnet/nnUNet_results"
```

### Activate environment (eg. conda)
```bash
conda activate seqseg
```

### Run
```bash
python3 auto_centerline.py --data_dir data --test_name 3d_fullres --train_dataset Dataset001_AORTAS --config_name global.yml --fold all --img_ext .nii.gz --outdir output --scale 1 --start 0 --stop -1 --max_n_steps 1000 --unit cm
```

### Details

`auto_centerline`: Main script to run.

Arguments:

-`data_dir`: This argument specifies the name of the folder containing the testing data (and test.json if applicable).

-`test_name`: This argument specifies the name of the nnUNet test to use. The default value is '3d_fullres'. Other possible values could be '2d', etc.

-`train_dataset`: This argument specifies the name of the dataset used to train the nnUNet model. For example, 'Dataset010_SEQCOROASOCACT'.

-'config_name': This argument specifies the name of the config file to use. The default value is 'global.yml'.

-`fold`: This argument specifies which fold to use for the nnUNet model. The default value is 'all'.

-`img_ext`: This argument specifies the image extension. For example, '.nii.gz'.

-`outdir`: This argument specifies the output directory where the results will be saved.

-`scale`: This argument specifies whether to scale image data. This is needed if the units for the nnUNet model and testing data are different. The default value is 1.

-`start`: This argument specifies where to start in the list of testing samples. The default value is 0.

-`stop`: This argument specifies where to stop in the list of testing samples. The default value is -1, which means to process all samples until the end of the list.

-`max_n_steps`: This argument specifies the maximum number of steps to run the algorithm. The default value is 1000.

-`unit`: This argument specifies the unit of the image data. The default value is 'cm'.

Data directory: Assumes the following structure:
- Directory
    - images
    - centerlines (if applicable)
    - truths (if applicable)
    - test_data.json (if applicable)

SeqSeg requires a seed point for initialization. This can be given by either:
- test.json file: located in data directory (see sample under data)
- centerline: if centerlines are given, we initialize using the first points of the centerline
- cardiac mesh: then the aortic valve must be labeled as Region 8 and LV 7

## Config file
`config/global.yml`: File contains config parameters, default is set but can be changed depending on task

We recommend duplicating the file and changing the name to avoid overwriting the default values.
If so, the config file must be passed as an argument when running the script: `config_name`

## Citation
When using SeqSeg, please cite the following [paper](https://rdcu.be/dU0wy):
    
```
@Article{SveinssonCepero2024,
author={Sveinsson Cepero, Numi
and Shadden, Shawn C.},
title={SeqSeg: Learning Local Segments for Automatic Vascular Model Construction},
journal={Annals of Biomedical Engineering},
year={2024},
month={Sep},
day={18},
issn={1573-9686},
doi={10.1007/s10439-024-03611-z},
url={https://doi.org/10.1007/s10439-024-03611-z},
}
```
