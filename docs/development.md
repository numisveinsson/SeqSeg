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

Install the optional training-data tools (vascular-segment-sampler):

```bash
pip install "seqseg[train]"
# or editable for local work:
pip install -e ../vascular-segment-sampler
pip install -e ".[train]"
```

Prepare patch datasets and convert to nnU-Net format:

```python
from vascular_segment_sampler.sampling import extract_patches
from vascular_segment_sampler.nnunet import write_nnunet_dataset

extract_patches(data_dir=..., outdir=..., config="global")
write_nnunet_dataset(indir=..., name="MYDATA", dataset_number=999, modality="ct")
```

Or via CLI: `vss-sample`, `vss-to-nnunet`.

1. **Prepare data** in nnUNet format (e.g. via `write_nnunet_dataset`):
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
seqseg run batch \
    -train_dataset Dataset999_MYCUSTOM \
    -fold 0 \
    -data_dir /path/to/data/ \
    -nnunet_results_path /path/to/nnUNet_results/ \
    -outdir results/
```

## Integration with Other Tools

### SimVascular Integration
With `-simvascular 1`, SeqSeg writes a full [SimVascular](http://simvascular.github.io/) project under each case:

```bash
seqseg run batch -data_dir data/ -outdir results/ -simvascular 1
# Open: results/{test_name}_{case}/simvascular/simvascular.proj
```

Layout:

| Path | Contents |
|------|----------|
| `simvascular.proj` | Project root file for SimVascular |
| `Images/` | Volume (`.vti`) and SV image sidecars |
| `Paths/` | Per-branch pathlines (`.pth`) |
| `Segmentations/` | Contour groups (`.ctgr`) paired with paths |
| `Models/` | Surface solid (`.vtp`) and companion (`.mdl`) |

To create or refresh the folder layout without re-running tracing:

```bash
seqseg simvascular init --case-dir results/3d_fullres_case_001/
```

See [Usage](usage.md#simvascular-project--simvascular-1) and the [tutorial](../seqseg/tutorial/tutorial.md#simvascular-integration) for the end-to-end CFD workflow.

### 3D Slicer Integration
```python
# Load SeqSeg results in 3D Slicer for visualization
import slicer
segmentation = slicer.util.loadSegmentation("result.mha")
```
