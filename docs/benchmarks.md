# Performance & Benchmarks

[← Back to README](../README.md)

## Aortic segmentation example

![Aortic segmentation: SeqSeg vs. 2D nnU-Net on the ATV dataset](https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/ATV_dataset_results.png)

*Qualitative comparison on 18 cases: ground truth, SeqSeg, 2D nnU-Net (with post-processing), and 2D nnU-Net raw predictions. SeqSeg tends to preserve a continuous aortic tree and peripheral branches where the 2D nnU-Net baselines are more fragmented or incomplete.*

## Coronary segmentation example

![Coronary artery segmentation with SeqSeg](https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/coronary.png)

*3D visualization: coronary tree segmented with SeqSeg (red) overlaid on the heart (transparent blue). Seed markers show the minimal initialization points used to grow the left and right coronary systems.*

## Performance Metrics

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
