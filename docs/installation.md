# Installation

[← Back to README](../README.md)

## System Requirements

- **OS**: Linux, macOS, Windows
- **Python**: ≥3.9 (3.11 recommended)
- **GPU**: CUDA-compatible GPU with ≥8GB VRAM (recommended for faster inference; can also run on CPU only)

## Option 1: pip Installation (Recommended)

```bash
pip install seqseg
seqseg --help  # Verify installation
```

## Option 2: Development Installation

```bash
git clone https://github.com/numisveinsson/SeqSeg.git
cd SeqSeg
pip install -e .
```

## Option 3: Conda Environment

```bash
conda create -n seqseg python=3.11
conda activate seqseg
pip install seqseg
```

## Dependencies

**Core Dependencies:**
```
nnunetv2                 # Deep learning segmentation
torch                    # PyTorch backend
SimpleITK                # Medical image I/O
vtk                      # 3D visualization and processing
PyYAML                   # Configuration management
scipy                    # Scientific computing
```

**Optional Dependencies:**
```
matplotlib               # Plotting and visualization
vmtk                     # Advanced vascular modeling tools
```

## Model Weights

Pre-trained weights are required for inference:

1. **Download**:
   - Aortic and femoral (MR/CT): [Pre-trained models](https://zenodo.org/records/15020477) (nnUNet_results folder)
   - Coronary CT (CCTA): [SeqSeg nnU-Net weights — CT coronary segmentation](https://zenodo.org/records/19547894) (`nnUNet_results_coronary.zip`, `Dataset010_SEQCOROASOCACT`)
2. **Extract**: Unzip to desired location
3. **Reference**: Use `-nnunet_results_path` to specify path

**Available Models:**
- `Dataset005_SEQAORTANDFEMOMR`: Aortic and femoral vessels (MR)
- `Dataset006_SEQAORTANDFEMOCT`: Aortic and femoral vessels (CT)
- `Dataset010_SEQCOROASOCACT`: Coronary lumen (CT angiography) — [nnU-Net weights on Zenodo](https://zenodo.org/records/19547894)
- Additional models for cerebral and pulmonary vessels available upon request
