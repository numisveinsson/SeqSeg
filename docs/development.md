# Research & Development

[← Back to README](../README.md)

Training a model on a new dataset is [Training](training.md). Tracking parameters are [Configuration](configuration.md).

## SimVascular

`seqseg run batch -simvascular 1` writes a [SimVascular](http://simvascular.github.io/) project under each case directory. Open `simvascular.proj` in that `simvascular/` folder.

The file layout is in [Usage](usage.md#simvascular-project--simvascular-1). The tutorial walks through opening the project and preparing it for CFD: [tutorial, SimVascular section](../seqseg/tutorial/tutorial.md#simvascular-integration).

To create or refresh the folder layout after a run that omitted the flag:

```bash
seqseg simvascular init --case-dir results/3d_fullres_case_001/
```

## 3D Slicer

Load the written files from the Slicer GUI:

1. **File → Add Data** and select the segmentation `.mha` (the volume named `{case}_segmentation_…`).
2. Add the surface `{case}_surface_mesh_….vtp` the same way.
3. In the Volumes or Models module, set the segmentation window/level or model visibility so the surface sits on the image.

## Repository layout

| Path | What it is |
|------|------------|
| `seqseg/cli.py` | Subcommands (`run`, `train`, `paths`, `doctor`, …) |
| `seqseg/pipeline/` | Batch tracing, the plus workflow, training, and post steps the CLI calls |
| `seqseg/modules/` | Patch extraction, nnU-Net prediction, centerlines, assembly |
| `seqseg/config/` | Packaged YAML tracking configs |
| `seqseg/tutorial/` | Sample aorta data and the tutorial |
| `docs/` | These guides |
