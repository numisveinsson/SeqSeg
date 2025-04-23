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
2. Load the medical image (`scan.nii.gz`) via **File > Open**.
3. Use the **Volume Rendering** option to inspect the image.
4. Identify and note the seed point coordinates for segmentation.

### Using VolView:
1. Open VolView.
2. Load the `scan.nii.gz` file.
3. Use the **Thresholding** and **Opacity Adjustments** to visualize structures.
4. Pick a seed point and record its coordinates.

## Running the Segmentation 🏃‍♂️🧠📈

Run the segmentation script `seqseg.py` with the required arguments: 🖥️🔢⚙️

```bash
python seqseg.py --num_steps <NUM_STEPS> \
                 --num_branches <NUM_BRANCHES> \
                 --seed_point <X,Y,Z> \
                 --out_dir <OUTPUT_DIRECTORY> \
                 --weights_path <WEIGHTS_PATH> \
                 --data_dir <DATA_DIRECTORY>
```

Example run with a given seed point (e.g., `100,120,80`):

```bash
python seqseg.py --num_steps 50 \
                 --num_branches 3 \
                 --seed_point 100,120,80 \
                 --out_dir results/ \
                 --weights_path models/weights.pth \
                 --data_dir data/
``` 🖥️🔢⚙️

```bash
python seqseg.py --num_steps <NUM_STEPS> \
                 --num_branches <NUM_BRANCHES> \
                 --seed_point <X,Y,Z> \
                 --out_dir <OUTPUT_DIRECTORY>
```

Example run with a given seed point (e.g., `100,120,80`):

```bash
python seqseg.py --num_steps 50 \
                 --num_branches 3 \
                 --seed_point 100,120,80 \
                 --out_dir results/
```

## Viewing the Output 📊🖼️🔬

After running the segmentation, the output will include: 🗂️📁✅
- `segmentation.vtp`: The segmented vessel structure.
- `centerline.vtp`: The extracted centerline of the vessels.
- `surface_mesh.vtp`: The reconstructed surface mesh.

To view the outputs in ParaView:
1. Open ParaView.
2. Load the `.vtp` files via **File > Open**.
3. Use the **Surface Rendering** mode for the surface mesh.
4. Apply the **Tube Filter** to visualize the centerline more clearly.

## Conclusion 🎯✅📌

This tutorial covers the end-to-end workflow for vessel segmentation. You should now be able to install the software, process medical images, and analyze results using ParaView. 🏥🧠📊

