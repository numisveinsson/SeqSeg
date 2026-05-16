# SeqSeg Tutorial: Vessel Segmentation Guide

A practical guide for automatic vessel segmentation using SeqSeg on medical images.

**SeqSeg 2.x:** This tutorial uses the structured CLI (`seqseg run batch`, etc.). Legacy invocations without a subcommand still work. For API and migration notes, see [What's new in 2.x](https://github.com/numisveinsson/SeqSeg#whats-new-in-2x) in the main README.

**Tutorial Dataset**: Abdominal aorta MR scan from SimVascular demo project  
**Expected Time**: 15-20 minutes  
**Expected Results**: Complete aortic segmentation with surface mesh and centerlines

## Quick Start Checklist

- [ ] Python 3.11+ installed
- [ ] SeqSeg package installed  
- [ ] Model weights downloaded (~0.2GB)
- [ ] Tutorial data available
- [ ] Seed points configured

**Success Criteria**: Generate segmentation, surface mesh, and centerlines for abdominal aorta

## 1. Installation

### Prerequisites
Ensure you have Python 3.11+ and Git installed:
```bash
python --version  # Should show 3.11+
git --version
```

### Install SeqSeg
```bash
# Create environment (recommended)
conda create -n seqseg python=3.11
conda activate seqseg

# Install SeqSeg
pip install seqseg

# Verify installation
seqseg --help
seqseg --version   # expect 2.0.0 or newer

# Optional: verify nnU-Net trainer folder after downloading weights
seqseg doctor --model-folder /path/to/nnUNet_results/Dataset005_SEQAORTANDFEMOMR/nnUNetTrainer__nnUNetPlans__3d_fullres
```

**✅ Checkpoint**: `seqseg --help` shows subcommands (`run`, `post`, `config`, `doctor`, `init`, …)

### Other useful 2.x commands (optional)

| Command | Purpose |
| -------- | ------- |
| `seqseg init dataset --path DIR` | Scaffold `images/`, `centerlines/`, `truths/`, template `seeds.json` |
| `seqseg run single` | One volume + seeds without hand-building the dataset tree |
| `seqseg config fingerprint --name global --baseline global_default` | Diff two packaged YAML configs |
| `from seqseg.api import run_tracing, …` | In-memory `sitk.Image` tracing (`disk_io=False`); see README |

### Newer CLI shortcuts (optional)

These commands complement the batch workflow in the rest of this tutorial:

- **`seqseg init dataset --path /path/to/dataset/`** — Creates `images/`, `centerlines/`, `truths/`, and a template `seeds.json` so you can drop volumes in and edit seeds before running `seqseg run batch`.
- **`seqseg run single`** — Runs one volume without manually building the full tree: it stages data under `<outdir>/_seqseg_single_staging/`. Use `--image`, `--outdir`, `--model-folder`, and either repeated **`--seed X Y Z R`** or **`--seeds-json`** (see `seqseg --help` after install).
- **`seqseg config fingerprint --name global --baseline global_default`** — Lists packaged YAML keys that differ from another config (useful after copying or editing configs).
- **`seqseg doctor --model-folder …/nnUNetTrainer__nnUNetPlans__3d_fullres`** — Checks imports and optionally verifies that the trainer directory exists.
- **Python:** high-level **`from seqseg.api import run_tracing, TracingOptions, branch_seed_at_point`** for in-memory `sitk.Image` workflows (see the main repository `README.md`).

## 2. Download Required Files

### Model Weights
Download pre-trained nnUNet weights (0.2GB):
```bash
# Download and extract
wget https://zenodo.org/records/15020477/files/nnUNet_results.zip
unzip nnUNet_results.zip
```
Alternative: [Manual download](https://zenodo.org/records/15020477)

### Tutorial Data
```bash
# Clone repository for tutorial data
git clone https://github.com/numisveinsson/SeqSeg.git
cd SeqSeg

# Verify data structure (SeqSeg expects lowercase images/ at runtime)
ls seqseg/tutorial/data/seeds.json
ls seqseg/tutorial/data/images/0110_0001.mha   # or Images/ on case-insensitive filesystems

# Linux: if only Images/ exists, add a symlink once:
#   cd seqseg/tutorial/data && ln -sf Images images
```

**Layout used by this tutorial:**

```
seqseg/tutorial/data/
├── images/0110_0001.mha    # volume (folder may appear as Images/ on macOS)
└── seeds.json              # seed points for case 0110_0001
```

**✅ Checkpoint**: Both model weights and tutorial data should be accessible

## 3. Understand the Data

### Image Overview
- **File**: `0110_0001.mha` (512×64×512 voxels, ~0.8,2,0.8 mm spacing)
- **Anatomy**: Abdominal aorta MR scan
- **Coordinates**: Physical coordinates in centimeters

### Seed Points Configuration
The `seeds.json` file contains initialization points:
```json
[
    {
        "name": "0110_0001",
        "seeds": [
            [
                [-2.07367, -2.1973, 13.4288],   # Start point
                [-1.17086, -1.33526, 12.2407],  # Direction point  
                1.1                              # Radius estimate (cm)
            ]
        ]
    }
]
```

**Format**: `[start_point, direction_point, radius_estimate]`

### Optional: Visualize Data
```bash
# View in ParaView (if installed)
paraview seqseg/tutorial/data/images/0110_0001.mha
```

**✅ Checkpoint**: Understand seed point format and data structure

## 4. Run Segmentation

### Basic command (recommended: `seqseg run batch`)

From the repository root, with `nnUNet_results` extracted next to the clone (adjust paths as needed):

**Linux/macOS:**
```bash
seqseg run batch \
    -data_dir seqseg/tutorial/data/ \
    -nnunet_results_path ../nnUNet_results/ \
    -nnunet_type 3d_fullres \
    -train_dataset Dataset005_SEQAORTANDFEMOMR \
    -fold all \
    -img_ext .mha \
    -config_name aorta_tutorial \
    -max_n_steps 10 \
    -max_n_branches 3 \
    -max_n_steps_per_branch 5 \
    -outdir tutorial_output/ \
    -unit cm \
    -scale 1 \
    -extract_global_centerline 1
```

**Windows (PowerShell):**
```powershell
seqseg run batch `
    -data_dir seqseg/tutorial/data/ `
    -nnunet_results_path ..\nnUNet_results\ `
    -nnunet_type 3d_fullres `
    -train_dataset Dataset005_SEQAORTANDFEMOMR `
    -fold all `
    -img_ext .mha `
    -config_name aorta_tutorial `
    -max_n_steps 10 `
    -max_n_branches 3 `
    -max_n_steps_per_branch 5 `
    -outdir tutorial_output\ `
    -unit cm `
    -scale 1 `
    -extract_global_centerline 1
```

**Windows (Command Prompt):** use the same flags with `^` line continuations and `seqseg run batch` instead of `seqseg` alone.

**Legacy (still supported):** omit `run batch` and pass the same flags — they are rewritten automatically.

### Alternative: single volume without dataset layout

If you only have one `.mha` and seed coordinates (no `seeds.json` tree):

```bash
seqseg run single \
  --image seqseg/tutorial/data/images/0110_0001.mha \
  --outdir tutorial_output_single/ \
  --model-folder ../nnUNet_results/Dataset005_SEQAORTANDFEMOMR/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --train-dataset Dataset005_SEQAORTANDFEMOMR \
  --config-name aorta_tutorial \
  --seed -2.07 -2.20 13.43 1.1 \
  --max-n-steps 10 --max-n-branches 3 --max-n-steps-per-branch 5
```

Staging is created under `tutorial_output_single/_seqseg_single_staging/`.

### Key arguments
| Argument | Purpose |
|----------|---------|
| `-data_dir` | Directory with `images/` (or `Images/`) and `seeds.json` |
| `-nnunet_results_path` | Path to extracted `nnUNet_results` folder |
| `-config_name` | Packaged YAML config (`aorta_tutorial` for this case) |
| `-max_n_steps` | Maximum tracking steps |
| `-max_n_branches` | Maximum branches to follow |
| `-max_n_steps_per_branch` | Max steps per branch |
| `-extract_global_centerline` | Write global centerline VTP when `1` |

### Expected Processing Time
- **Initialization**: ~30 seconds
- **Segmentation**: ~2-4 minutes  
- **Post-processing**: ~1 minute
- **Total**: ~3-5 minutes

**✅ Checkpoint**: Process completes without errors
## 5. Analyze Results

### Output files

After a successful run (with `-max_n_steps 10` and `-nnunet_type 3d_fullres`), main artifacts are under `tutorial_output/`:

```
tutorial_output/
├── 0110_0001_segmentation_3d_fullres_10_steps.mha   # Final binary segmentation
├── 0110_0001_surface_mesh_3d_fullres_10_steps.vtp   # Smoothed surface mesh
├── 0110_0001_centerline_3d_fullres_10_steps.vtp     # Global centerline (if -extract_global_centerline 1)
└── 3d_fullres_0110_0001/                            # Per-case working directory
    ├── out.txt
    ├── simvascular/                                 # SimVascular project (paths, contours, …)
    └── …                                            # Intermediate VTK/MHA if -write_steps 1
```

Step counts in filenames match the actual number of tracing steps taken (here, 10 if the run completes as configured).

### Quick quality check
```bash
# Check segmentation volume (should be ~15-25 cm³ for aorta)
python -c "
import SimpleITK as sitk
import numpy as np
seg = sitk.ReadImage('tutorial_output/0110_0001_segmentation_3d_fullres_10_steps.mha')
volume = np.sum(sitk.GetArrayFromImage(seg)) * np.prod(seg.GetSpacing()) / 1000
print(f'Segmented volume: {volume:.1f} cm³')
"
```

### Visualization
```bash
paraview tutorial_output/0110_0001_surface_mesh_3d_fullres_10_steps.vtp
paraview tutorial_output/0110_0001_centerline_3d_fullres_10_steps.vtp
```

**✅ Checkpoint**: Output files exist and volume is reasonable (15-25 cm³)

## 6. Troubleshooting

### Common Issues

**"No seeds found"**
- Check `seeds.json` format and file location
- Verify case name matches image filename

**Poor segmentation quality**
```bash
# Increase steps and branches (keep other flags from section 4)
seqseg run batch \
    -data_dir seqseg/tutorial/data/ \
    -nnunet_results_path ../nnUNet_results/ \
    -outdir tutorial_output/ \
    -img_ext .mha \
    -config_name aorta_tutorial \
    -max_n_steps 20 \
    -max_n_branches 5
```

**Import or nnU-Net path errors**
```bash
seqseg doctor
seqseg doctor --model-folder ../nnUNet_results/Dataset005_SEQAORTANDFEMOMR/nnUNetTrainer__nnUNetPlans__3d_fullres
```

### Debug mode
For detailed analysis:
```bash
seqseg run batch \
    -write_steps 1 \
    -max_n_steps 3 \
    # ... same paths as in section 4
```
Intermediate files appear under `tutorial_output/3d_fullres_0110_0001/` in `volumes/`, `predictions/`, `centerlines/`, `surfaces/`, etc.

## Next Steps

### SimVascular Integration

#### Step 1: Import SeqSeg Results
1. **Open SimVascular**
2. **Create New Project**: 
   - File → New SV Project → Choose location
   - Name: `SeqSeg_Aorta_Tutorial`

3. **Import Surface Model**:
   - Right-click **Models** tab → **Import Solid Model**
   - Select: `tutorial_output/0110_0001_surface_mesh_3d_fullres_10_steps.vtp`
   - Name: `AortaModel_SeqSeg`
   - Or open the generated project: `tutorial_output/3d_fullres_0110_0001/simvascular/`
   - **Import** → Should see model in 3D view

#### Step 2: Model Preparation
1. **Select Model**: Click `AortaModel_SeqSeg` in Models tab

2. **Extract Faces**:
   - **Face Ops** tab → **Extract Faces**
   - Separation Angle: `90.0` (high value for single face)
   - **Extract Faces** → Creates face list

3. **Clean Model** (if needed):
   - **Global Ops** tab:
     - **Smooth**: Iterations=10, Relaxation=0.01
     - **Decimate**: Target reduction=0.1
     - **Remesh**: Edge size=0.5mm

#### Step 3: Prepare for CFD
1. **Trim Model** (create inlets/outlets):
   - **Local Ops** tab → **Trim**
   - Select cutting plane location
   - **Trim** to create flat inlet/outlet faces

2. **Fill Holes**:
   - **Face Ops** → **Fill Holes w IDs**
   - Assigns face IDs for boundary conditions

3. **Label Faces**:
   - Inlet: Face ID 1
   - Outlet: Face ID 2  
   - Wall: Face ID 3

#### Step 4: Mesh Generation
1. **Create Mesh**:
   - **Meshes** tab → **Create Mesh**
   - Select model: `AortaModel_SeqSeg`
   - Mesh parameters:
     - Global Edge Size: `0.5` mm
     - Surface Mesh Flag: `1`
     - Volume Mesh Flag: `1`

2. **Generate Mesh**: **Run Mesher**

#### Step 5: Simulation Setup
1. **Create Simulation**:
   - **Simulations** tab → **Create Simulation Job**
   - Solver: **svSolver**

2. **Boundary Conditions**:
   - **Inlet** (Face 1): Prescribed velocities
   - **Outlet** (Face 2): RCR boundary condition
   - **Wall** (Face 3): No-slip

3. **Material Properties**:
   - Density: `1.06 g/cm³`
   - Viscosity: `0.04 Poise`

#### Step 6: Run Simulation
1. **Pre-processor**: Generate solver input files
2. **Run Simulation**: Execute CFD solver
3. **Post-processing**: Analyze results in ParaView

### Custom data

1. **Scaffold a dataset (optional)**:
   ```bash
   seqseg init dataset --path your_data/
   # Copy volumes into your_data/images/
   # Edit your_data/seeds.json
   ```

2. **Or prepare manually**:
   ```
   your_data/
   ├── images/
   │   └── your_case.mha
   └── seeds.json
   ```

3. **Create `seeds.json`**:
   ```json
   [
       {
           "name": "your_case",
           "seeds": [
               [[x1, y1, z1], [x2, y2, z2], radius_estimate]
           ]
       }
   ]
   ```

4. **Run SeqSeg**:
   ```bash
   seqseg run batch -data_dir your_data/ -nnunet_results_path ../nnUNet_results/ -outdir results/ -img_ext .mha -config_name global
   ```

### Python API (optional)

With seeds and config already chosen, embed tracing without writing SeqSeg outputs:

```python
import SimpleITK as sitk
from seqseg.api import BranchSeed, TracingOptions, run_tracing

image = sitk.ReadImage("seqseg/tutorial/data/images/0110_0001.mha")
result = run_tracing(
    image,
    [BranchSeed(
        old_point=[-2.07367, -2.1973, 13.4288],
        new_point=[-1.17086, -1.33526, 12.2407],
        radius=1.1,
    )],
    "/path/to/nnUNet_results/Dataset005_SEQAORTANDFEMOMR/nnUNetTrainer__nnUNetPlans__3d_fullres",
    config="aorta_tutorial",
    options=TracingOptions(disk_io=False, unit="cm", max_n_steps=10),
)
prob = result.assembly.assembly  # sitk.Image; threshold for binary mask
```

### Additional Resources
- **SeqSeg Documentation**: [GitHub Repository](https://github.com/numisveinsson/SeqSeg)
- **SimVascular Tutorial**: [SimVascular.org](https://simvascular.github.io/)
- **nnUNet Documentation**: [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
- **ParaView User Guide**: [ParaView.org](https://www.paraview.org/documentation/)

**Need help?** Check [GitHub Issues](https://github.com/numisveinsson/SeqSeg/issues) or [documentation](https://github.com/numisveinsson/SeqSeg)

---

**Tutorial Complete!** You should now have a segmented aorta ready for downstream analysis.