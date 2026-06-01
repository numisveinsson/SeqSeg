# Algorithm Overview

[← Back to README](../README.md)

## Methodology

SeqSeg employs a **sequential tracking approach** that combines:

1. **Local CNN Segmentation**: nnU-Net provides probabilistic segmentation of local 3D patches
2. **Geometric Tracking**: Vessel-specific tracking algorithm follows centerlines and detects bifurcations
3. **Iterative Refinement**: Sequential processing builds complete vascular trees from seed points

## Technical Workflow

![SeqSeg Workflow](https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/seqseg.png)

**Step-by-step process:**
1. **Initialization**: Place seed points manually or from prior centerlines
2. **Local Segmentation**: Extract and segment 3D patches using trained nnU-Net
3. **Centerline Extraction**: Compute local vessel centerlines and orientations
4. **Step Planning**: Determine next position along vessel direction
5. **Bifurcation Detection**: Identify and queue branch points
6. **Iteration**: Repeat until vessel termination or max steps reached

## Training Strategy

![Training Pipeline](https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/seqseg_training.png)

The neural network is trained on **local subvolume patches** extracted from annotated vessel datasets, enabling:
- **Generalization** across different vessel scales and orientations
- **Efficient training** with limited annotated data
- **Real-time inference** on standard GPUs

## Key Algorithmic Innovations

- **Adaptive patch sizing**: Automatically adjusts to vessel diameter
- **Multi-scale processing**: Handles vessels from 1mm to 30mm diameter
- **Topology preservation**: Maintains vessel connectivity during segmentation
- **Branch prioritization**: Intelligent exploration of vessel trees
