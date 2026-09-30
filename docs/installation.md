# Installation

[← Back to README](../README.md)

## System Requirements

- **OS**: Linux, macOS, Windows ([Windows install steps](../seqseg/tutorial/windows.md))
- **Python**: 3.9 or newer (3.11 recommended; that is the version in the tutorial)
- **GPU**: CUDA-compatible GPU with ≥8GB VRAM (faster inference). CPU-only runs are supported.

## Option 1: pip Installation (Recommended)

```bash
pip install seqseg
seqseg --help  # Verify installation
```

To prepare training data and train new nnU-Net models for SeqSeg:

```bash
pip install "seqseg[train]"   # adds vascular-segment-sampler
seqseg paths init             # saves ~/nnunet_data/... in ~/.seqseg/paths.yaml
seqseg doctor                 # checks sampler + saved/effective paths
```

See [Training](training.md).

## Option 2: Development Installation

```bash
git clone https://github.com/numisveinsson/SeqSeg.git
cd SeqSeg
pip install -e .
pip install -e ".[train]"   # optional: training-data tools
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
matplotlib                       # Plotting and visualization (pip install "seqseg[viz]")
vascular-segment-sampler         # Patch sampling + nnU-Net dataset prep (pip install "seqseg[train]")
vmtk                             # Advanced vascular modeling tools
```

`vascular-segment-sampler` is required for `seqseg train prepare`. See [Training](training.md).

## Model Weights

Inference needs a downloaded nnU-Net results folder:

```bash
curl -L -o nnUNet_results.zip https://zenodo.org/records/15020477/files/nnUNet_results.zip
unzip nnUNet_results.zip
```

After unzip, the layout looks like this:

```
nnUNet_results/
└── Dataset005_SEQAORTANDFEMOMR/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/    # weights live here
```

`seqseg run batch -nnunet_results_path` points at `nnUNet_results/`. `seqseg run single --model-folder` and `seqseg doctor --model-folder` point at the inner `nnUNetTrainer__nnUNetPlans__3d_fullres` directory.

| Weights | Anatomy | Pass to `seqseg run batch` |
|---------|---------|----------------------------|
| [Zenodo 15020477](https://zenodo.org/records/15020477) `Dataset005_SEQAORTANDFEMOMR` | Aorta and femoral, MR | `-train_dataset Dataset005_SEQAORTANDFEMOMR -config_name global_aorta` |
| [Zenodo 15020477](https://zenodo.org/records/15020477) `Dataset006_SEQAORTANDFEMOCT` | Aorta and femoral, CT | `-train_dataset Dataset006_SEQAORTANDFEMOCT -config_name global_aorta` |
| [Zenodo 19547894](https://zenodo.org/records/19547894) `Dataset010_SEQCOROASOCACT` (`nnUNet_results_coronary.zip`) | Coronary lumen, CT angiography | `-train_dataset Dataset010_SEQCOROASOCACT -config_name global_coro` |

The [tutorial](../seqseg/tutorial/tutorial.md) uses `Dataset005_SEQAORTANDFEMOMR` with `-config_name aorta_tutorial` on its sample MR scan. Cerebral and pulmonary weights are available on request; their configs are `global_cereb` and `global_pulm`.

Check a trainer folder after download:

```bash
seqseg doctor --model-folder nnUNet_results/Dataset005_SEQAORTANDFEMOMR/nnUNetTrainer__nnUNetPlans__3d_fullres
```
