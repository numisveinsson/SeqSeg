# SeqSeg Segmentation Tutorial ✨🩺💻

This tutorial will guide you through installing the required software, downloading necessary resources, and running the segmentation pipeline on a medical image. 🏥📊🖥️

This tutorial is based on the SimVascular tutorial, using the same MR medical image scan. The SimVascular project is under `data/`

## Installation 🚀🔧🐍

First, install the required dependencies using pip or conda. 📦🔗
Simply,
```bash
pip install seqseg
```
Test the installation by running the following command in your terminal: 🖥️✅

```bash
seqseg --help
```
If the installation was successful, you should see the help message for the SeqSeg command-line interface. 📜💬

## Download Neural Network Weights 📥🤖

Download the pre-trained neural network weights from the following link: 🔗⬇️💾

[Download Weights](<https://zenodo.org/records/15020477>)

Note that the model was trained on a dataset of images in centimeters. If you are running inference on images in millimeters, you will need to specify the `-unit mm` and `-scale 0.1` flag when running the segmentation script. 📏🔍

You will need to specify the path to the downloaded directory `nnUNet_results` when running the segmentation script. 🛤️

## Download Tutorial Data 📂

You need to clone the SeqSeg repository from GitHub to get the tutorial data and scripts. This repository contains the medical image and seed points.
```bash
git clone https://github.com/numisveinsson/SeqSeg.git
```
Note that this data is the Demo Project from the SimVascular project, you can find more information about the project [here](https://simvascular.github.io/). 📚🌐

## Viewing the tutorial Medical Image 🔍🖥️🩻

You can visualize the medical image in ParaView or VolView to identify a suitable seed point. 🎯🔬👀

The medical image is located in the `data/Images` directory and is named `0110_0001.mha`. This image is a 3D volume of a MR scan of a human abdominal aorta. 🧠🩻

Hint: if you do not wish to choose a seed point, one is given below.

### Using ParaView:
1. Open ParaView.
2. Load the medical image (`0110_0001.mha`) from the `data/Images` directory.
3. Use the **Volume Rendering** option to inspect the image.
4. Identify and note the seed point coordinates for segmentation.

### Using VolView:
1. Open VolView.
2. Load the `0110_0001.mha` image from the `data/Images` directory.
3. Use the **Thresholding** and **Opacity Adjustments** to visualize structures.
4. Pick a seed point and record its coordinates.

## Running the Segmentation 🏃‍♂️🧠📈

1. Define seed point coordinates in the ``data/seeds.json`` file. 📍📝

The file has default seed points you can use or you can specify your own. The coordinates should be in the format `[x, y, z]` and correspond to the physical coordinates in the 3D image, see the file for example. Note that each seed point requires ``two`` coordinates: to define a vector direction for the initialization. The ``third`` argument is a radius estimate, usually for aortas a radius=1.1cm is a good enough approximation. 🧭

Example `seeds.json` file:

```json
[
    {
        "name": "0110_0001",
        "seeds": [
            [   [-2.07367, -2.1973, 13.4288] , [-1.17086, -1.33526, 12.2407] , 1.1  ]
        ]
    }
]
```

2. Run the segmentation script `seqseg` with the required arguments: 🖥️🔢⚙️

```bash
seqseg \
    -data_dir seqseg/tutorial/data/ \
    -nnunet_results_path .../nnUNet_results/ \
    -test_name 3d_fullres \
    -train_dataset Dataset005_SEQAORTANDFEMOMR \
    -fold all \
    -img_ext .mha \
    -config_name aorta_tutorial \
    -max_n_steps 5 \
    -max_n_branches 2 \
    -outdir output/
    -unit cm \
    -scale 1 \
    -start 0 \
    -stop 1
```
### Explanation of Arguments: 📜🔍

- `-data_dir`: Path to the directory containing the medical images and `seeds.json` file. 📂
- `-nnunet_results_path`: Path to the directory containing the pre-trained nnUNet model weights. 📁
- `-test_name`: Name of the nnUNet test configuration to use (e.g., `3d_fullres`). 📊
- `-train_dataset`: Name of the dataset used to train the nnUNet model (e.g., `Dataset005_SEQAORTANDFEMOMR`). 📚
- `-fold`: Specifies which fold to use for the nnUNet model (e.g., `0` or `all` for all folds). 📅
- `-img_ext`: File extension of the medical images (e.g., `.mha`). 📸
- `-config_name`: Name of the configuration file to use (e.g., `aorta_tutorial`). 📄
- `-max_n_steps`: Maximum number of steps for the segmentation algorithm (e.g., `5`). ⏳
- `-max_n_branches`: Maximum number of branches to explore during segmentation (e.g., `2`). 🌿
- `-outdir`: Directory where the output files will be saved (e.g., `output/`). 📂
- `-unit`: Unit of measurement for the coordinates (e.g., `cm`). 📏
- `-scale`: Scale factor for the coordinates (e.g., `1`). 📐
- `-start`: Starting index for processing images (e.g., `0`). 🔢
- `-stop`: Stopping index for processing images (e.g., `1` to process only the first image). 🔚

### Debugging: 🐞🔍
If you encounter any issues, you can set the `-write_samples` flag to `1` to write out the samples used for training. This can help in debugging the segmentation process. 🛠️🔧

```bash
seqseg \
    -write_samples 1 \
    ...
```
This will create folders `samples` in the output directory, containing the output from each step. 📂🗂️

## Viewing the Output 📊🖼️🔬

After running the segmentation, the output will include: 🗂️📁✅
- `segmentation.mha`: The segmented vessel structure.
- `centerline.vtp`: The extracted centerline of the vessels.
- `surface_mesh.vtp`: The reconstructed surface mesh.

To view the outputs in ParaView:
1. Open ParaView.
2. Load the `.vtp` files via **File > Open**.
3. Use the **Surface Rendering** mode for the surface mesh.
4. Apply the **Tube Filter** to visualize the centerline more clearly.

## Conclusion 🎯✅📌

This tutorial covers the end-to-end workflow for vessel segmentation. You should now be able to install the software, process medical images, and analyze results using ParaView. 🏥🧠📊

