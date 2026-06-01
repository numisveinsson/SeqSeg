# Research & Development

[← Back to README](../README.md)

## Extending SeqSeg

### Custom Neural Networks
```python
# Replace nnUNet with custom segmentation model
from seqseg.modules.prediction import CustomPredictor

class MyPredictor(CustomPredictor):
    def predict_patch(self, image_patch):
        # Implement custom prediction logic
        return segmentation_prediction
```

## Training New Models

To train nnUNet models on custom datasets:

1. **Prepare data** in nnUNet format:
```bash
nnUNet_raw/Dataset999_MYCUSTOM/
├── imagesTr/          # Training images
├── labelsTr/          # Training labels
├── imagesTs/          # Test images (optional)
└── dataset.json       # Dataset metadata
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

## Integration with Other Tools

### SimVascular Integration
SeqSeg outputs are compatible with [SimVascular](http://simvascular.github.io/) for CFD modeling:
```bash
# Output .vtp files can be directly imported into SimVascular
# for mesh generation and flow simulation
```
SeqSeg also provides a `simvascular/Paths/` directory with pre-formatted path files.

### 3D Slicer Integration
```python
# Load SeqSeg results in 3D Slicer for visualization
import slicer
segmentation = slicer.util.loadSegmentation("result.mha")
```
