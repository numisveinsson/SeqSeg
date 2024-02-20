# SeqSeg: Automatic Tracing and Segmentation of Blood Vessels
Repository for Automatic Vascular Model Creation using localized 3D segmentation for tracing.

Arguments:

-data_dir: This argument specifies the name of the folder containing the testing data.

-test_name: This argument specifies the name of the nnUNet test to use. The default value is '3d_fullres'. Other possible values could be '2d', etc.

-dataset: This argument specifies the name of the dataset used to train the nnUNet model. For example, 'Dataset010_SEQCOROASOCACT'.

-fold: This argument specifies which fold to use for the nnUNet model. The default value is 'all'.

-img_ext: This argument specifies the image extension. For example, '.nii.gz'.

-outdir: This argument specifies the output directory where the results will be saved.

-scale: This argument specifies whether to scale image data. This is needed if the units for the nnUNet model and testing data are different. The default value is 1.

-start: This argument specifies where to start in the list of testing samples. The default value is 0.

-stop: This argument specifies where to stop in the list of testing samples. The default value is -1, which usually means to process all samples until the end of the list.

## Set Up
Main package dependencies:
    nnU-Net
    Pytorch
    SITK
    VTK
    VMTK
(see environment.yml file)

## Running
auto_centerline: Main file to run.

Data directory: Assumes the following structure:
    - images
    - centerlines (if applicable)
    - truths (if applicable)

SeqSeg requires a seed point for initialization. This can be given by either:
    - test.json file: located in data directory (see sample under data)
    - centerline: if centerlines are given, we initialize using the first points of the centerline

