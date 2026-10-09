# Usage

[← Back to README](../README.md)

For a complete walkthrough with example data, see the [step-by-step tutorial](../seqseg/tutorial/tutorial.md). Weight downloads and which dataset pairs with which config are in [Installation](installation.md).

## Commands and flag spelling

`seqseg run batch` and `seqseg run plus batch` take underscore flags (`-outdir` / `--outdir`). The data folder is `-data_dir` or `--data_directory`. `seqseg run single`, `seqseg train`, and `seqseg paths` take hyphen flags (`--image`, `--data-dir`). `seqseg train prepare` also accepts `-data_dir`.

A 1.x command that starts with a flag (`seqseg -data_dir ...`) still runs. It is rewritten to `seqseg run batch`.

`-img_ext` is required on every batch command. These three can be omitted when they are saved with `seqseg paths`:

| Flag | Role |
|------|------|
| `-data_dir` / `--data_directory` | Folder with `images/` and `seeds.json` |
| `-outdir` | Where results are written |
| `-nnunet_results_path` | Folder that contains `Dataset00…/` (see [Installation](installation.md)) |

`-train_dataset` defaults to `Dataset010_SEQCOROASOCACT` and `-config_name` defaults to `global`. Set both to the pair for your anatomy:

| Anatomy | `-train_dataset` | `-config_name` |
|---------|------------------|----------------|
| Aorta / femoral MR | `Dataset005_SEQAORTANDFEMOMR` | `global_aorta` |
| Aorta / femoral CT | `Dataset006_SEQAORTANDFEMOCT` | `global_aorta` |
| Coronary CT | `Dataset010_SEQCOROASOCACT` | `global_coro` |
| Cerebral | weights on request | `global_cereb` |
| Pulmonary | weights on request | `global_pulm` |

The tutorial sample uses `-config_name aorta_tutorial` with the aorta MR weights.

## Data Preparation

### Directory Structure
```
your_project/
├── images/              # Medical images (.nii.gz, .mha, .nrrd)
├── seeds.json           # Seed point coordinates
├── centerlines/         # Optional: existing centerlines
└── truths/              # Optional: ground truth segmentations
```

You can scaffold this layout with:

```bash
seqseg init dataset --path your_project/
```

### Supported Image Formats
- **NIfTI**: `.nii`, `.nii.gz`
- **MetaImage**: `.mha`, `.mhd`
- **NRRD**: `.nrrd`
- **DICOM**: Via SimpleITK readers
- **Others**: Any [SimpleITK-supported format](https://simpleitk.readthedocs.io/en/master/IO.html)

### Seed points

`seeds.json` sits next to `images/`. The case `name` plus `-img_ext` is the filename: `case_001` with `-img_ext .mha` loads `images/case_001.mha`, and the same name with `-img_ext .nii.gz` loads `images/case_001.nii.gz`.

Each seed is three values:

1. **Start point** — `[x, y, z]` where tracing begins
2. **Direction point** — a second point further along the vessel, so SeqSeg knows which way to walk
3. **Radius** — approximate lumen radius at the start

Coordinates and that radius use `-unit` (default `cm`). They are physical coordinates, the same space as the image header.

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

Reading the inner list: start `[-2.07, -2.20, 13.43]`, direction `[-1.17, -1.34, 12.24]`, radius `1.1` cm. Add another three-item list inside `"seeds"` for a second branch (for example the other coronary ostium).

Radius guesses that work as a starting point, in the default centimeter unit:

- Coronary lumen: `0.2` (2 mm)
- Aortic root: `1.1` (11 mm)

YAML tracking thresholds such as `MIN_RADIUS` and `STOP_RADIUS` stay in millimeters even when `-unit cm`. See [Configuration](configuration.md).

Other ways to initialize:

- **Existing centerlines** in `centerlines/`: tracing can start from the first points (`-num_seeds_centerline`, `-pt_centerline`)
- **Cardiac meshes**: aortic valve (region 8) and LV (region 7) labels

### Units

`-unit` (`cm` or `mm`) is the unit of the seed coordinates and the seed radius. It should match the image header. `-scale` multiplies the voxel spacing passed to nnU-Net. Use `-scale 0.1` with `-unit mm` when the image spacing is in millimeters and the model was trained with spacing in centimeters.

## Basic Usage

`seqseg run batch` on an aorta MR case:

```bash
seqseg run batch \
    -data_dir /path/to/data/ \
    -nnunet_results_path /path/to/nnUNet_results/ \
    -nnunet_type 3d_fullres \
    -train_dataset Dataset005_SEQAORTANDFEMOMR \
    -fold all \
    -img_ext .mha \
    -config_name global_aorta \
    -outdir results/ \
    -simvascular 1
```

Other common commands:

| Command | Purpose |
| -------- | ------- |
| `seqseg run single` | One volume + seeds (stages under `<outdir>/_seqseg_single_staging/`) |
| `seqseg run plus batch` | Global nnU-Net sweep, then SeqSeg |
| `seqseg paths init` / `set` / `show` | Save default nnU-Net / data / out dirs (`~/.seqseg/paths.yaml`) |
| `seqseg train prepare` | Extract patches or whole volumes and create an nnU-Net Dataset (`pip install "seqseg[train]"`) |
| `seqseg train nnunet` | Plan/preprocess and train with nnU-Net |
| `seqseg doctor` | Check imports and optional trainer folder |
| `seqseg post global-centerline` | Centerlines from an existing segmentation (all bodies by default) |
| `seqseg simvascular init` | Create/refresh a SimVascular project layout under a case directory |

Training a model on a new dataset is documented in [Training](training.md).

### Centerlines from an existing segmentation

```bash
seqseg post global-centerline single --seg case.mha --out case_centerline.vtp
```

```bash
seqseg post global-centerline batch --seg-dir results/ --seg-glob "*.mha"
```

With no `--seeds-json`, one seed is placed in each disconnected body. Pass `--seeds-json` and `--case-name` to cap tracing at that many bodies (largest first).

## Advanced Usage Examples

#### SimVascular project export:
```bash
seqseg run batch -data_dir data/ -outdir results/ -simvascular 1
```

#### Debugging mode (write out intermediate results):
```bash
seqseg run batch -data_dir data/ -max_n_steps 100 -max_n_branches 10 -write_steps 1
```

#### Batch processing:
```bash
seqseg run batch -data_dir data/ -start 0 -stop 50  # Process cases 0-49
```

#### Scale adjustment:
```bash
seqseg run batch -data_dir data/ -unit mm -scale 0.1  # Model trained in cm, data in mm
```

#### Start from an existing segmentation:
```bash
seqseg run batch -data_dir data/ -start_seg /path/to/initial_seg.mha
```
Tracing is unchanged; the SeqSeg result is merged (union) into the initial mask.

## Command Line Arguments

Arguments for `seqseg run batch` (same flags as legacy flat CLI):

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_dir` | str | required | Path to data directory containing images and seeds.json. Saved `seqseg paths` value is used when omitted |
| `nnunet_results_path` | str | required | Folder that contains `Dataset00…/` weight directories. Saved `seqseg paths` value is used when omitted |
| `nnunet_type` | str | `3d_fullres` | nnUNet model architecture (`3d_fullres`, `2d`) |
| `train_dataset` | str | `Dataset010_SEQCOROASOCACT` | Weight folder name. Change this to match the anatomy (see the table above) |
| `fold` | str | `all` | Cross-validation fold (`all`, `0`, `1`, `2`, `3`, `4`) |
| `img_ext` | str | required | Image file extension (`.nii.gz`, `.mha`, `.nrrd`) |
| `config_name` | str | `global` | Packaged YAML name without `.yaml`. Pair it with `train_dataset` |
| `outdir` | str | required | Output directory for results. Saved `seqseg paths` value is used when omitted |
| `unit` | str | `cm` | Image coordinate units (`mm`, `cm`) |
| `scale` | float | `1.0` | Multiplies voxel spacing passed to nnU-Net. `0.1` converts mm spacing to cm |
| `max_n_steps` | int | `1000` | Maximum tracking steps. The tutorial sets this to `10` so the demo finishes quickly |
| `max_n_steps_per_branch` | int | `100` | Maximum steps per vessel branch |
| `max_n_branches` | int | `100` | Maximum number of branches to follow |
| `start` | int | `0` | Starting case index for batch processing |
| `stop` | int | `-1` | Ending case index (-1 for all) |
| `write_steps` | int | `0` | Save intermediate results (0/1) |
| `extract_global_centerline` | int | `0` | Extract final centerline (0/1) |
| `cap_surface_cent` | int | `0` | Cap vessel surface ends (0/1) |
| `pt_centerline` | int | `50` | Centerline point spacing for seed extraction |
| `num_seeds_centerline` | int | `1` | Number of seeds for centerline initialization |
| `start_seg` | str | - | Optional initial segmentation; SeqSeg output is merged into it |
| `simvascular` | int | `0` | Write SimVascular project under each case (`0`/`1`) |

## Output Files

SeqSeg generates several output files for each processed case. Filenames include `{test_name}` (e.g. `3d_fullres`):

| File | Description |
|------|-------------|
| `{case}_segmentation_{test_name}_{steps}_steps.mha` | Final binary segmentation |
| `{case}_surface_mesh_{test_name}_{steps}_steps.vtp` | Smoothed 3D surface mesh |
| `{case}_centerline_{test_name}_{steps}_steps.vtp` | Extracted vessel centerlines (only when `extract_global_centerline=1`) |
| `{case}_binary_seg_*.mha` | Raw binary segmentation |
| `{case}_prob_seg_*.mha` | Probabilistic segmentation |

Per-case working directory (e.g. `results/3d_fullres_{case}/`):

**For debugging** (when `write_steps=1`):
- `volumes/`: Local image patches
- `predictions/`: nnUNet predictions
- `centerlines/`: Intermediate centerlines
- `surfaces/`: Intermediate surfaces
- `points/`: Tracking points

### SimVascular project (`-simvascular 1`)

When enabled, each case folder also contains a ready-to-open SimVascular project:

```
{test_name}_{case}/simvascular/
├── simvascular.proj          # Open this in SimVascular
├── Images/{case}.vti         # Volume (+ sidecars for SV)
├── Paths/*.pth               # Pathlines per branch
├── Segmentations/*.ctgr      # Contour groups per path
└── Models/{case}.vtp         # Surface solid (+ companion .mdl)
```

Open `simvascular.proj` (or the `simvascular/` folder) in SimVascular to load the image, paths, contours, and model together. You can also scaffold or refresh a project layout later with:

```bash
seqseg simvascular init --case-dir results/3d_fullres_case_001/
```
