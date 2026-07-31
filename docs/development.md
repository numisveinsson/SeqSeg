# Research & Development

[← Back to README](../README.md)

To train SeqSeg models on a new dataset, see [Training](training.md).

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
