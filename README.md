![example workflow](https://github.com/numisveinsson/SeqSeg/actions/workflows/python-app.yml/badge.svg)
![example workflow](https://github.com/numisveinsson/SeqSeg/actions/workflows/test.yml/badge.svg)

# SeqSeg: Sequential Vessel Segmentation and Tracking

> **Automatic tracking and segmentation of blood vessels in CT and MR images using deep learning and geometric tracking**

[![Paper](https://img.shields.io/badge/Paper-Annals%20of%20BME-blue)](https://rdcu.be/dU0wy)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)

## Abstract

SeqSeg is a novel method for automatic vessel segmentation that combines local deep learning predictions with geometric tracking algorithms. The approach segments vessels sequentially by taking steps along vessel centerlines and detecting bifurcations, enabling robust segmentation of complex vascular trees from minimal initialization.

**Key Features:**
- **Minimal supervision**: Requires only 1-2 seed points for initialization
- **Robust bifurcation detection**: Automatically identifies and follows vessel branches  
- **Multi-modal support**: Works with CT and MR 3D medical imaging modalities
- **Scalable**: Handles vessels from small coronaries to large aortas
- **Clinical validation**: Tested on diverse vascular anatomies including coronary, aortic, cerebral, and pulmonary vessels (only aortic weights released)

**Performance Highlights:**
- Dice similarity coefficient: >0.9 on validation datasets
- Processing time: ~2-10 minutes per case (depending on vessel complexity)
- Computational requirements: Standard CPU and GPU (for faster inference) hardware

![SeqSeg Demo](seqseg/assets/mr_model_tracing_fast_shorter.gif)
*Real-time demonstration: Automatic segmentation of abdominal aorta in 3D MR scan*

## Table of Contents

- [Quick Start](#quick-start)
- [Algorithm Overview](#algorithm-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Performance & Benchmarks](#performance--benchmarks)
- [Configuration](#configuration)
- [Research & Development](#research--development)
- [Citation](#citation)

## Quick Start

For immediate use with pre-trained models:

```bash
# Install
pip install seqseg

# Download weights (see tutorial for links)
# Run segmentation
seqseg -data_dir your_data/ -nnunet_results_path path/to/weights/ -config_name aorta_tutorial
```

**📖 Complete Tutorial**: [Step-by-step guide](https://github.com/numisveinsson/SeqSeg/blob/main/seqseg/tutorial/tutorial.md) with example data and detailed instructions.

## Algorithm Overview

### Methodology

SeqSeg employs a **sequential tracking approach** that combines:

1. **Local CNN Segmentation**: nnU-Net provides probabilistic segmentation of local 3D patches
2. **Geometric Tracking**: Vessel-specific tracking algorithm follows centerlines and detects bifurcations
3. **Iterative Refinement**: Sequential processing builds complete vascular trees from seed points

### Technical Workflow

![SeqSeg Workflow](seqseg/assets/seqseg.png)

**Step-by-step process:**
1. **Initialization**: Place seed points manually or from prior centerlines
2. **Local Segmentation**: Extract and segment 3D patches using trained nnU-Net
3. **Centerline Extraction**: Compute local vessel centerlines and orientations
4. **Step Planning**: Determine next position along vessel direction
5. **Bifurcation Detection**: Identify and queue branch points
6. **Iteration**: Repeat until vessel termination or max steps reached

### Training Strategy

![Training Pipeline](seqseg/assets/seqseg_training.png)

The neural network is trained on **local subvolume patches** extracted from annotated vessel datasets, enabling:
- **Generalization** across different vessel scales and orientations
- **Efficient training** with limited annotated data
- **Real-time inference** on standard GPUs

### Key Algorithmic Innovations

- **Adaptive patch sizing**: Automatically adjusts to vessel diameter
- **Multi-scale processing**: Handles vessels from 1mm to 30mm diameter
- **Topology preservation**: Maintains vessel connectivity during segmentation
- **Branch prioritization**: Intelligent exploration of vessel trees

## Installation

### System Requirements

- **OS**: Linux, macOS, Windows
- **Python**: ≥3.9 (3.11 recommended)
- **GPU**: CUDA-compatible GPU with ≥8GB VRAM (recommended for faster inference; can also run on CPU only)

### Option 1: pip Installation (Recommended)

```bash
pip install seqseg
seqseg --help  # Verify installation
```

### Option 2: Development Installation

```bash
git clone https://github.com/numisveinsson/SeqSeg.git
cd SeqSeg
pip install -e .
```

### Option 3: Conda Environment

```bash
conda create -n seqseg python=3.11
conda activate seqseg
pip install seqseg
```

### Dependencies

**Core Dependencies:**
```
nnunetv2                 # Deep learning segmentation
torch                    # PyTorch backend  
SimpleITK                # Medical image I/O
vtk                      # 3D visualization and processing
PyYAML                   # Configuration management
scipy                    # Scientific computing
```

**Optional Dependencies:**
```
matplotlib               # Plotting and visualization
vmtk                    # Advanced vascular modeling tools
```

### Model Weights

Pre-trained weights are required for inference:

1. **Download**: [Pre-trained Models](https://zenodo.org/records/15020477) (nnUNet_results folder)
2. **Extract**: Unzip to desired location
3. **Reference**: Use `-nnunet_results_path` to specify path

**Available Models:**
- `Dataset005_SEQAORTANDFEMOMR`: Aortic and femoral vessels (MR)
- `Dataset006_SEQAORTANDFEMOCT`: Aortic and femoral vessels (CT)
- Additional models available for coronary, cerebral, and pulmonary vessels upon request

## Usage

### Data Preparation

#### Directory Structure
```
your_project/
├── images/              # Medical images (.nii.gz, .mha, .nrrd)
├── seeds.json          # Seed point coordinates  
├── centerlines/        # Optional: existing centerlines
└── truths/            # Optional: ground truth segmentations
```

#### Supported Image Formats
- **NIfTI**: `.nii`, `.nii.gz`
- **MetaImage**: `.mha`, `.mhd`  
- **NRRD**: `.nrrd`
- **DICOM**: Via SimpleITK readers
- **Others**: Any [SimpleITK-supported format](https://simpleitk.readthedocs.io/en/master/IO.html)

#### Seed Point Specification

Seeds can be provided via:

1. **JSON file** (recommended):
```json
[
    {
        "name": "case_001",
        "seeds": [
            [[-2.07, -2.20, 13.43], [-1.17, -1.34, 12.24], 1.1]
        ]
    }
]
```
Format: `[[start_point], [direction_point], radius_estimate]`

2. **Existing centerlines**: Automatic initialization from first points
3. **Cardiac meshes**: Aortic valve (Region 8) and LV (Region 7) labels

### Basic Usage

```bash
seqseg \
    -data_dir /path/to/data/ \
    -nnunet_results_path /path/to/nnUNet_results/ \
    -nnunet_type 3d_fullres \
    -train_dataset Dataset005_SEQAORTANDFEMOMR \
    -fold all \
    -img_ext .mha \
    -config_name aorta_tutorial \
    -outdir results/
```

### Advanced Usage Examples

#### Debugging mode (write out intermediate results):
```bash
seqseg -data_dir data/ -max_n_steps 100 -max_n_branches 10 -write_steps 1
```

#### Batch processing:
```bash
seqseg -data_dir data/ -start 0 -stop 50  # Process cases 0-49
```

#### Scale adjustment:
```bash
seqseg -data_dir data/ -unit mm -scale 0.1  # Model trained in cm, data in mm
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_dir` | str | - | Path to data directory containing images and seeds.json |
| `nnunet_results_path` | str | - | Path to nnUNet model weights directory |
| `nnunet_type` | str | `3d_fullres` | nnUNet model architecture (`3d_fullres`, `2d`) |
| `train_dataset` | str | `Dataset010_SEQCOROASOCACT` | Dataset name used for training (e.g., `Dataset005_SEQAORTANDFEMOMR`) |
| `fold` | str | `all` | Cross-validation fold (`all`, `0`, `1`, `2`, `3`, `4`) |
| `img_ext` | str | - | Image file extension (`.nii.gz`, `.mha`, `.nrrd`) |
| `config_name` | str | `global` | Configuration file name |
| `outdir` | str | - | Output directory for results |
| `unit` | str | `cm` | Image coordinate units (`mm`, `cm`) |
| `scale` | float | `1.0` | Scaling factor for unit conversion |
| `max_n_steps` | int | `1000` | Maximum tracking steps |
| `max_n_steps_per_branch` | int | `100` | Maximum steps per vessel branch |
| `max_n_branches` | int | `100` | Maximum number of branches to follow |
| `start` | int | `0` | Starting case index for batch processing |
| `stop` | int | `-1` | Ending case index (-1 for all) |
| `write_steps` | int | `0` | Save intermediate results (0/1) |
| `extract_global_centerline` | int | `0` | Extract final centerline (0/1) |
| `cap_surface_cent` | int | `0` | Cap vessel surface ends (0/1) |
| `pt_centerline` | int | `50` | Centerline point spacing for seed extraction |
| `num_seeds_centerline` | int | `1` | Number of seeds for centerline initialization |

### Output Files

SeqSeg generates several output files for each processed case. Filenames include `{test_name}` (e.g. `3d_fullres`):

| File | Description |
|------|-------------|
| `{case}_segmentation_{test_name}_{steps}_steps.mha` | Final binary segmentation |
| `{case}_surface_mesh_{test_name}_{steps}_steps.vtp` | Smoothed 3D surface mesh |
| `{case}_centerline_{test_name}_{steps}_steps.vtp` | Extracted vessel centerlines (only when `extract_global_centerline=1`) |
| `{case}_binary_seg_*.mha` | Raw binary segmentation |
| `{case}_prob_seg_*.mha` | Probabilistic segmentation |
| `simvascular/` | Directory with SimVascular-compatible files |

**For debugging** (when `write_steps=1`):
- `volumes/`: Local image patches
- `predictions/`: nnUNet predictions  
- `centerlines/`: Intermediate centerlines
- `surfaces/`: Intermediate surfaces
- `points/`: Tracking points

## Performance & Benchmarks

### Performance Metrics

**Processing Times** (Local CPU, typical cases):
- Simple vessel (aorta): ~2-5 minutes
- Complex tree (coronary): ~5-15 minutes  
- Full cerebral vasculature: ~10-30 minutes

**Accuracy** (validation on held-out test sets):
- Dice Similarity Coefficient: >0.9
- Hausdorff Distance: <35 pixels
- Centerline accuracy: >0.9

**Scalability**:
- Tested on images up to 512³ voxels
- Handles vessel diameters from 1mm to 30mm
- Supports vessel trees with 50+ branches

## Configuration

### Configuration Files

SeqSeg uses YAML configuration files located in `seqseg/config/`:

| Config File | Purpose |
|-------------|---------|
| `global.yaml` | Default settings |
| `aorta_tutorial.yaml` | Aortic vessel segmentation |
| `global_coro.yaml` | Coronary arteries |
| `global_cereb.yaml` | Cerebral vessels |
| `global_pulm.yaml` | Pulmonary vessels |

### Key Configuration Parameters

```yaml
# Volume extraction
VOLUME_SIZE_RATIO: 5              # Local volume size vs radius (4.9 for aorta, 5.5 for coronaries)
MAGN_RADIUS: 1                    # Radius magnification factor
ADD_RADIUS: 0.3                   # Additional radius for volume extraction (mm)
MIN_RADIUS: 0.3                   # Minimum vessel radius before stopping (mm)

# Tracing control
NR_CHANCES: 2                     # Retry attempts for failed steps
NR_ALLOW_RETRACE_STEPS: 5         # Steps allowed inside existing vessels before stopping
PREVENT_RETRACE: True             # Avoid tracing already segmented areas
ASSEMBLY_EVERY_N: 20              # Combine predictions into assembly every N steps

# Early stopping
STOP_PRE: True                    # Enable premature stopping
STOP_RADIUS: 0.46                 # Stop tracing if radius drops below this (mm)
MAX_STEPS_BRANCH: 1000            # Max steps per branch

# Centerline extraction
CENTERLINE_EXTRACTION_VMTK: False # Use VMTK (True) or built-in FMM (False) for centerlines
```

### Custom Configuration

1. Copy existing config: `cp seqseg/config/global.yaml seqseg/config/my_config.yaml`
2. Modify parameters for your specific use case
3. Run with: `seqseg -config_name my_config ...`

## Research & Development

### Extending SeqSeg

#### Custom Neural Networks
```python
# Replace nnUNet with custom segmentation model
from seqseg.modules.prediction import CustomPredictor

class MyPredictor(CustomPredictor):
    def predict_patch(self, image_patch):
        # Implement custom prediction logic
        return segmentation_prediction
```

### Training New Models

To train nnUNet models on custom datasets:

1. **Prepare data** in nnUNet format:
```bash
nnUNet_raw/Dataset999_MYCUSTOM/
├── imagesTr/          # Training images  
├── labelsTr/          # Training labels
├── imagesTs/          # Test images (optional)
└── dataset.json      # Dataset metadata
```

2. **Train model**:
```bash
nnUNetv2_plan_and_preprocess -d 999
nnUNetv2_train 999 3d_fullres 0  # Train fold 0
```

3. **Use with SeqSeg**:
```bash
seqseg -train_dataset Dataset999_MYCUSTOM -fold 0
```

### Integration with Other Tools

#### SimVascular Integration
SeqSeg outputs are compatible with [SimVascular](http://simvascular.github.io/) for CFD modeling:
```bash
# Output .vtp files can be directly imported into SimVascular
# for mesh generation and flow simulation
```
SeqSeg also provides a `simvascular/Paths/` directory with pre-formatted path files.

#### 3D Slicer Integration  
```python
# Load SeqSeg results in 3D Slicer for visualization
import slicer
segmentation = slicer.util.loadSegmentation("result.mha")
```

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
