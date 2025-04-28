# SeqSeg Segmentation Tutorial ✨🩺💻

This tutorial will guide you through installing the required software, downloading necessary resources, and running the segmentation pipeline on a medical image. 🏥📊🖥️

This tutorial is based on the SimVascular tutorial, using the same MR medical image scan. The SimVascular project is under `data/`

## Installation 🚀🔧🐍

First, install the required dependencies using pip or conda. See main ReadMe for details. 📦🔗

## Download Neural Network Weights 📥🤖📂

Download the pre-trained neural network weights from the following link: 🔗⬇️💾

[Download Weights](<https://zenodo.org/records/15020477>)

Save the weights folder called nnUNet_results in a directory, e.g., `models/`. 📁🗂️💾

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

1. Define where the weights (downloaded above) are located. 📂🔍

```bash
export nnUNet_results="../example/path/to/nnUNet_results"
```

2. Define seed point coordinates in the ``data/seeds.json`` file. 📍📝

The file has default seed points you can use or you can specify your own. The coordinates should be in the format `[x, y, z]` and correspond to the physical coordinates in the 3D image, see the file for example. Note that each seed point requires ``two`` coordinates: to define a vector direction for the initialization. The ``third`` argument is a radius estimate, usually for aortas a radius=1.1cm is a good enough approximation. 🧭

3. Run the segmentation script `seqseg.py` with the required arguments: 🖥️🔢⚙️

```bash
python seqseg.py \
    -test_name 3d_fullres \
    -train_dataset Dataset001_AORTAS \
    -data_dir data/ \
    -img_ext .mha \
    -config_name global_aorta \
    -max_n_steps <NUM_STEPS> \
    -max_n_branches <NUM_BRANCHES> \
    -outdir <OUTPUT_DIRECTORY> \
```

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

