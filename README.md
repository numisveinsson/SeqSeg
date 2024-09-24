# SeqSeg: Automatic Tracing and Segmentation of Blood Vessels
Repository for Automatic Vascular Model Creation using localized 3D segmentation for tracing.

Here is a gif showing the output of the algorithm on a 3D MR image of the descending aorta:

![](assets/mr_model_tracing_fast_shorter.gif)

## Set Up
Main package dependencies (see environment.yml file for all):
- SITK
- VTK

and if using nnU-Net:
- nnU-Net
- Pytorch

and if using VMTK:
- VMTK

## Config file
`config/global.yml`: File contains config parameters, default is set but can be changed depending on task

We recommend duplicating the file and changing the name to avoid overwriting the default values.
If so, the config file must be passed as an argument when running the script: `config_name`

## Running
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

