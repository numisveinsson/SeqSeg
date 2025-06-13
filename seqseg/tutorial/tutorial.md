# SeqSeg Segmentation Tutorial ✨

This tutorial will guide you through installing the required software, downloading necessary resources, and running the segmentation pipeline on a medical image. 🏥

This tutorial is based on the SimVascular tutorial, using the same MR medical image scan. The SimVascular project is under `data/`

## Installation 🚀

First, install the required dependencies using pip or conda. 📦

Best practice is to create a virtual environment for the installation. You can use `venv` or `conda` for this purpose. See the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for more information on creating virtual environments.

Simply,
```bash
pip install seqseg
```
Test the installation by running the following command in your terminal: 🖥️

```bash
seqseg --help
```
If the installation was successful, you should see the help message for the SeqSeg command-line interface.

## Download Neural Network Weights 📥

Download the pre-trained neural network weights from the following link: 🔗⬇

[Download Weights](<https://zenodo.org/records/15020477>)

Note that the model was trained on a dataset of images in centimeters. If you are running inference on images in millimeters, you will need to specify the `-unit mm` and `-scale 0.1` flag when running the segmentation script.

You will need to specify the path to the downloaded directory `nnUNet_results` when running the segmentation script.
- For example, you can place the downloaded directory in the same directory as the `SeqSeg` cloned repository, and specify the path as `-nnunet_results_path ../nnUNet_results/`. See the example below for more details. 📂

Note: make sure to unzip the downloaded file.
- On windows, you can use the built-in unzip functionality by right-clicking the file and selecting "Extract All".
- On macOS, you can double-click the file to unzip it.

## Download Tutorial Data 📂

You need to clone the SeqSeg repository from GitHub to get the tutorial data and scripts. This repository contains the medical image and seed points.
```bash
git clone https://github.com/numisveinsson/SeqSeg.git
```
Note that this data is the Demo Project from the SimVascular project, you can find more information about the project [here](https://simvascular.github.io/). 📚🌐

## Viewing the tutorial Medical Image 🔍

You can visualize the medical image in ParaView or VolView to identify a suitable seed point. 🎯

The medical image is located in the `data/Images` directory and is named `0110_0001.mha`. This image is a 3D volume of a MR scan of a human abdominal aorta. 🧠

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

## Running the Segmentation Pipeline 💻

1. Define seed point coordinates in the ``data/seeds.json`` file. 📍

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

2. Run the segmentation script `seqseg` with the required arguments: 🖥️

Make sure to be inside the `SeqSeg` directory, and the following command will run the segmentation pipeline on the medical image using the pre-trained nnUNet model. The output will be saved in the `output/` directory that is created automatically. 📂

MacOS/Linux:
```bash
seqseg \
    -data_dir seqseg/tutorial/data/ \
    -nnunet_results_path ../nnUNet_results/ \
    -nnunet_type 3d_fullres \
    -train_dataset Dataset005_SEQAORTANDFEMOMR \
    -fold all \
    -img_ext .mha \
    -config_name aorta_tutorial \
    -max_n_steps 5 \
    -max_n_branches 2 \
    -outdir output/ # This will be created automatically 📂
    -unit cm \
    -scale 1 \
    -start 0 \
    -stop 1 \
    -write_steps 0 \
    -extract_global_centerline 0

```
Windows:
```bash
seqseg `
    -data_dir seqseg/tutorial/data/ `
    -nnunet_results_path ..\nnUNet_results\ `
    -nnunet_type 3d_fullres `
    -train_dataset Dataset005_SEQAORTANDFEMOMR `
    -fold all `
    -img_ext .mha `
    -config_name aorta_tutorial `
    -max_n_steps 5 `
    -max_n_branches 2 `
    -outdir output/ `
    -unit cm `
    -scale 1 `
    -start 0 `
    -stop 1 `
    -write_steps 0 `
    -extract_global_centerline 0
```
### Explanation of Arguments: 📜🔍

- `-data_dir`: Path to the directory containing the medical images and `seeds.json` file. 📂
- `-nnunet_results_path`: Path to the directory containing the pre-trained nnUNet model weights. 📁
- `-nnunet_tpe`: Type of nnUNet model to use (e.g., `3d_fullres` or `2d`). 📊
- `-train_dataset`: Name of the dataset used to train the nnUNet model (e.g., `Dataset005_SEQAORTANDFEMOMR`). 📚
- `-fold`: Specifies which fold to use for the nnUNet model (e.g., `0` or `all` for all folds). 📅
- `-img_ext`: File extension of the medical images (e.g., `.mha`). 📸
- `-config_name`: Name of the configuration file to use (e.g., `aorta_tutorial`). 📄
- `-max_n_steps`: Maximum number of steps for the segmentation algorithm (e.g., `5`). ⏳
- `-max_n_branches`: Maximum number of branches to explore during segmentation (e.g., `2`). 🌿
- `-outdir`: Directory where the output files will be saved (e.g., `output/`). (This will be created automatically) 📂
- `-unit`: Unit of measurement for the coordinates (e.g., `cm`). 📏
- `-scale`: Scale factor for the coordinates (e.g., `1`). 📐
- `-start`: Starting index for processing images (e.g., `0`). 🔢
- `-stop`: Stopping index for processing images (e.g., `1` to process only the first image). 🔚
- `-write_steps`: If set to `1`, writes out all the steps. Useful for debugging. (default is `0`)
- `-extract_global_centerline`: If set to `1`, extracts the global centerline of the segmented vessels. (default is `0`)

### Debugging: 🐞🔍
If you encounter any issues, you can set the `-write_samples` flag to `1` to write out the samples used for training. This can help in debugging the segmentation process. 🛠️🔧

```bash
seqseg \
    -write_steps 1 \
    ...
```
This will create folders with all the intermediate steps in the output directory, allowing you to inspect the segmentation process step-by-step. 📂🔍

- `volumes` : contains the subvolumes extracted from the original image.
- 'surfaces` : contains the surfaces extracted from the segmentation predictions.
- `centerlines` : contains the centerlines extracted from the segmentation predictions.
- `predictions` : contains the segmentation predictions.
- `points` : contains the points chosen to move to the next step.

## Viewing the Output 📊🔬

After running the segmentation, the output will include: 🗂️✅
- `0110_0001_seg_containing_seeds_X_steps.mha`: The segmented vessel structure.
- `0110_0001_centerline_X_steps.vtp`: The extracted centerline of the vessels.
- `0110_0001_surface_mesh_smooth_X_steps.vtp`: The reconstructed surface mesh.

where the `X` in the filenames corresponds to the number of steps taken in the segmentation process.

The `0110_0001` prefix corresponds to the name of the medical image used for segmentation.

To view the outputs in ParaView:
1. Open ParaView.
2. Load the `.vtp` files via **File > Open**.
3. Use the **Surface Rendering** mode for the surface mesh.
4. Apply the **Tube Filter** to visualize the centerline more clearly.

## Conclusion 🎯✅

This tutorial covers the end-to-end workflow for vessel segmentation. You should now be able to install the software, process medical images, and analyze results using ParaView. 🏥🧠📊

## Importing and Modeling in SimVascular 🛠️

To import the segmented vessel into SimVascular for further modeling:
1. Open SimVascular.
2. Right click the `Models` tab and select `Import Solid Model`.
3. Select the `0110_0001_surface_mesh_smooth_X_steps.vtp` file from the output directory.
4. Name the model (e.g., `AortaModel_SeqSeg`).
4. Click on the `AortaModel_SeqSeg` model in the `Models` tab to select it.
5. Under `Face Ops`, select `Extract Faces` and choose Seqaration Angle as `90.0`. (High so we get one face for the whole aorta)
6. Click `Extract Faces` to extract the faces.
7. Now you can use `Global Ops` to
    - `Smooth` the model
    - `Decimate` the model
    - `Remesh` the model
8. Or use `Local Ops` to
    - `Trim` the model to create inlets and outlets
    - `Smooth` the model locally
9. Once you have trimmed the model, you can use `Face Ops` to
    - `Fill Holes w IDs` to fill the trimmed holes in the model
    - Label the faces of the model for boundary conditions
    10. Model is ready for meshing and simulation in SimVascular! 🎉