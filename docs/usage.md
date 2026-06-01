# Usage

[← Back to README](../README.md)

For a complete walkthrough with example data, see the [step-by-step tutorial](../seqseg/tutorial/tutorial.md).

## Data Preparation

### Directory Structure
```
your_project/
├── images/              # Medical images (.nii.gz, .mha, .nrrd)
├── seeds.json           # Seed point coordinates
├── centerlines/         # Optional: existing centerlines
└── truths/              # Optional: ground truth segmentations
```

### Supported Image Formats
- **NIfTI**: `.nii`, `.nii.gz`
- **MetaImage**: `.mha`, `.mhd`
- **NRRD**: `.nrrd`
- **DICOM**: Via SimpleITK readers
- **Others**: Any [SimpleITK-supported format](https://simpleitk.readthedocs.io/en/master/IO.html)

### Seed Point Specification

Seeds can be provided via:

Typical seed point radius estimates:
- Coronary vessels: `0.2 cm` (`2 mm`)
- Aortic root: `1.1 cm` (`11 mm`)

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

## Basic Usage

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

## Advanced Usage Examples

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

## Command Line Arguments

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

## Output Files

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
