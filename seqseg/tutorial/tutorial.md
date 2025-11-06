# SeqSeg Tutorial: Vessel Segmentation Guide

A practical guide for automatic vessel segmentation using SeqSeg on medical images.

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
```

**✅ Checkpoint**: Command should display SeqSeg help information

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

# Verify data structure
ls seqseg/tutorial/data/Images/  # Should show: 0110_0001.mha
ls seqseg/tutorial/data/         # Should show: seeds.json
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
paraview seqseg/tutorial/data/Images/0110_0001.mha
```

**✅ Checkpoint**: Understand seed point format and data structure

## 4. Run Segmentation

### Basic Command

**Linux/macOS:**
```bash
seqseg \
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
seqseg `
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

**Windows (Command Prompt):**
```cmd
seqseg ^
    -data_dir seqseg/tutorial/data/ ^
    -nnunet_results_path ..\nnUNet_results\ ^
    -nnunet_type 3d_fullres ^
    -train_dataset Dataset005_SEQAORTANDFEMOMR ^
    -fold all ^
    -img_ext .mha ^
    -config_name aorta_tutorial ^
    -max_n_steps 10 ^
    -max_n_branches 3 ^
    -max_n_steps_per_branch 5 ^
    -outdir tutorial_output\ ^
    -unit cm ^
    -scale 1 ^
    -extract_global_centerline 1
```

### Key Arguments
| Argument | Purpose |
|----------|---------|
| `-data_dir` | Path to images and seeds.json |
| `-nnunet_results_path` | Path to model weights |
| `-max_n_steps` | Maximum tracking steps |
| `-max_n_branches` | Maximum branches to follow |
| `-max_n_steps_per_branch` | Max steps per branch |
| `-extract_global_centerline` | Generate centerlines (1=yes) |

### Expected Processing Time
- **Initialization**: ~30 seconds
- **Segmentation**: ~2-4 minutes  
- **Post-processing**: ~1 minute
- **Total**: ~3-5 minutes

**✅ Checkpoint**: Process completes without errors
## 5. Analyze Results

### Output Files
After successful completion, you'll find:
```
tutorial_output/
├── 0110_0001_seg_containing_seeds_10_steps.mha     # Binary segmentation
├── 0110_0001_surface_mesh_smooth_10_steps.vtp      # 3D surface mesh
├── 0110_0001_centerline_10_steps.vtp               # Vessel centerlines
└── out.txt                                         # Processing log
```

### Quick Quality Check
```bash
# Check segmentation volume (should be ~15-25 cm³ for aorta)
python -c "
import SimpleITK as sitk
import numpy as np
seg = sitk.ReadImage('tutorial_output/0110_0001_seg_containing_seeds_10_steps.mha')
volume = np.sum(sitk.GetArrayFromImage(seg)) * np.prod(seg.GetSpacing()) / 1000
print(f'Segmented volume: {volume:.1f} cm³')
"
```

### Visualization
```bash
# View results in ParaView
paraview tutorial_output/0110_0001_surface_mesh_smooth_10_steps.vtp

# View centerlines
paraview tutorial_output/0110_0001_centerline_10_steps.vtp
```

**✅ Checkpoint**: Output files exist and volume is reasonable (15-25 cm³)

## 6. Troubleshooting

### Common Issues

**"No seeds found"**
- Check `seeds.json` format and file location
- Verify case name matches image filename

**Poor segmentation quality**
```bash
# Increase steps and branches
seqseg \
    -max_n_steps 20 \
    -max_n_branches 5 \
    # ... other arguments
```

### Debug Mode
For detailed analysis:
```bash
seqseg \
    -write_steps 1 \
    -max_n_steps 3 \
    # ... other arguments
```
This creates intermediate files in: `volumes/`, `predictions/`, `centerlines/`, `surfaces/`

## Next Steps

### SimVascular Integration

#### Step 1: Import SeqSeg Results
1. **Open SimVascular**
2. **Create New Project**: 
   - File → New SV Project → Choose location
   - Name: `SeqSeg_Aorta_Tutorial`

3. **Import Surface Model**:
   - Right-click **Models** tab → **Import Solid Model**
   - Select: `tutorial_output/0110_0001_surface_mesh_smooth_10_steps.vtp`
   - Name: `AortaModel_SeqSeg`
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

### Custom Data
To use your own images:
1. **Prepare Data Structure**:
   ```
   your_data/
   ├── Images/
   │   └── your_case.mha
   └── seeds.json
   ```

2. **Create Seeds File**:
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

3. **Run SeqSeg**:
   ```bash
   seqseg -data_dir your_data/ -nnunet_results_path ../nnUNet_results/ ...
   ```

### Additional Resources
- **SeqSeg Documentation**: [GitHub Repository](https://github.com/numisveinsson/SeqSeg)
- **SimVascular Tutorial**: [SimVascular.org](https://simvascular.github.io/)
- **nnUNet Documentation**: [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
- **ParaView User Guide**: [ParaView.org](https://www.paraview.org/documentation/)

**Need help?** Check [GitHub Issues](https://github.com/numisveinsson/SeqSeg/issues) or [documentation](https://github.com/numisveinsson/SeqSeg)

---

**Tutorial Complete!** You should now have a segmented aorta ready for downstream analysis.