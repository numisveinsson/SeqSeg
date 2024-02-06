# SeqSeg: Automatic Tracing and Segmentation of Blood Vessels
Repository for Automatic Vascular Model Creation using localized 3D segmentation for tracing.

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

