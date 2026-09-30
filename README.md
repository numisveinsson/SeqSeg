![example workflow](https://github.com/numisveinsson/SeqSeg/actions/workflows/python-app.yml/badge.svg)
![example workflow](https://github.com/numisveinsson/SeqSeg/actions/workflows/test.yml/badge.svg)

<p align="center">
  <img src="https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/seqseg_logo.png" alt="SeqSeg — Sequential Vessel Segmentation" width="480"/><br/>
  <img src="https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/coronary.png" alt="Example coronary segmentation (SeqSeg)" width="260"/>
</p>

<h1 align="center">SeqSeg: Sequential Vessel Segmentation and Tracking</h1>

<p align="center">
  <b>Automatic tracking and segmentation of blood vessels in CT and MR images using deep learning and geometric tracking.</b>
</p>

<p align="center">
  <a href="https://rdcu.be/dU0wy"><img src="https://img.shields.io/badge/Paper-Annals%20of%20BME-blue" alt="Paper"/></a>
  <a href="https://github.com/numisveinsson/SeqSeg/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"/></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+"/></a>
</p>

> **News:** SeqSeg writes a full SimVascular project in the `simvascular/` subdirectory — open it directly in SimVascular with pathlines and contours for every segmented branch.

---

## Why SeqSeg?

SeqSeg segments vessels **sequentially**, taking steps along vessel centerlines and detecting bifurcations to grow complete vascular trees from just **1–2 seed points**. By combining local deep-learning predictions (nnU-Net) with geometric tracking, it stays robust across vessel scales — from small coronaries to large aortas.

- 🌱 **Minimal supervision** — only 1–2 seed points to initialize
- 🌿 **Robust bifurcation detection** — automatically follows every branch
- 🩻 **Multi-modal** — works with CT and MR 3D medical images
- 📏 **Scalable** — vessels from ~1mm coronaries to ~30mm aortas
- ✅ **Clinically validated** — coronary, aortic, cerebral, and pulmonary anatomies (pre-trained weights for aorta CT/MR and coronary CT)
- ⚡ **Fast** — on CPU, about 2–5 minutes for an aorta, 5–15 for a coronary tree, and 10–30 for a cerebral tree. GPU is faster. Details in [Benchmarks](docs/benchmarks.md).

<p align="center">
  <img src="https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/mr_model_tracing_fast_shorter.gif" alt="SeqSeg Demo"/><br/>
  <i>Real-time demonstration: automatic segmentation of an abdominal aorta in a 3D MR scan.</i>
</p>

## What's new in 2.x

SeqSeg **2.0** uses subcommands. A command from 1.x that starts with a flag, such as `seqseg -data_dir ...`, still runs: it is rewritten to `seqseg run batch` automatically.

### Commands

| Command | Description |
| -------- | ----------- |
| **`seqseg run batch`** | Dataset batch tracing |
| **`seqseg run single`** | One volume and seeds. Stages files under `<outdir>/_seqseg_single_staging/`, then traces |
| **`seqseg run plus batch`** | Global nnU-Net sweep, then SeqSeg |
| **`seqseg init dataset`** | Create `images/`, `centerlines/`, `truths/`, and a template `seeds.json` |
| **`seqseg paths init` / `set` / `show`** | Save default nnU-Net, data, and output directories in `~/.seqseg/paths.yaml` |
| **`seqseg train prepare`** | Extract patches or whole volumes and build an nnU-Net dataset (`pip install "seqseg[train]"`) |
| **`seqseg train nnunet`** | Run nnU-Net planning, preprocessing, and training |
| **`seqseg doctor`** | Check imports (SimpleITK, vtk, nnunetv2, scipy, optional sampler) and paths |
| **`seqseg config dump` / `fingerprint`** | Print a packaged YAML config, or list keys that differ from another |
| **`seqseg post global-centerline`** | Build global centerlines from existing segmentations |
| **`seqseg simvascular init`** | Create or refresh a SimVascular project under a case directory |
| **`seqseg --version`** | Print the installed version |

`seqseg run batch` flags use underscores (`-outdir` / `--outdir`). The data folder is `-data_dir` or `--data_directory`. `seqseg run single`, `seqseg train`, and `seqseg paths` use hyphens (`--image`, `--data-dir`).

### Migrating from 1.x

1. Prefer `seqseg run batch`. The old flag-only form still works.
2. Use `seqseg run plus batch` in place of `python -m seqseg.seqseg_plus`. The nnU-Net path flags are unchanged.
3. Check the install with `seqseg --version` after `pip install -U seqseg`.

## Quick Start

Python 3.9 or newer. 3.11 is the version used in the tutorial and conda example.

```bash
pip install seqseg

# Aorta and femoral weights (MR and CT). Coronary weights are a separate zip; see Installation.
curl -L -o nnUNet_results.zip https://zenodo.org/records/15020477/files/nnUNet_results.zip
unzip nnUNet_results.zip
```

`-nnunet_results_path` is the folder that contains `Dataset005_SEQAORTANDFEMOMR` (and the other dataset folders in that zip).

**First run:** follow the [step-by-step tutorial](seqseg/tutorial/tutorial.md). It includes an abdominal-aorta MR scan, seed points, and Windows notes. The tutorial caps the number of steps so the example finishes in a few minutes; a real case should use the defaults.

**Your own data** needs `images/` and `seeds.json` (see [Usage](docs/usage.md)). This command is an aorta MR case. Change `-train_dataset`, `-config_name`, and `-img_ext` to match your images and weights:

```bash
seqseg run batch \
    -data_dir your_data/ \
    -nnunet_results_path nnUNet_results/ \
    -train_dataset Dataset005_SEQAORTANDFEMOMR \
    -config_name global_aorta \
    -img_ext .mha \
    -outdir results/
```

| Anatomy | `-train_dataset` | `-config_name` |
|---------|------------------|----------------|
| Aorta / femoral MR | `Dataset005_SEQAORTANDFEMOMR` | `global_aorta` |
| Aorta / femoral CT | `Dataset006_SEQAORTANDFEMOCT` | `global_aorta` |
| Coronary CT | `Dataset010_SEQCOROASOCACT` | `global_coro` |

The tutorial uses `-config_name aorta_tutorial` for its sample scan. Download links and the trainer-folder layout are in [Installation](docs/installation.md).

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Setup, dependencies, and pre-trained model weights |
| [Tutorial](seqseg/tutorial/tutorial.md) | End-to-end aorta example with sample data |
| [Usage](docs/usage.md) | Data layout, seeds, CLI arguments, and output files |
| [Configuration](docs/configuration.md) | YAML configs and tracking parameters |
| [Algorithm Overview](docs/algorithm.md) | Methodology, workflow, and training strategy |
| [Training](docs/training.md) | Train nnU-Net models on a new dataset |
| [Performance & Benchmarks](docs/benchmarks.md) | Accuracy, timing, and qualitative comparisons |
| [API](docs/api.md) | Call tracing from Python |
| [Research & Development](docs/development.md) | SimVascular, 3D Slicer, and repository layout |

## Citation

When using SeqSeg, please cite the following [paper](https://rdcu.be/dU0wy):

```
@Article{SveinssonCepero2024,
author={Sveinsson Cepero, Numi
and Shadden, Shawn C.},
title={SeqSeg: Learning Local Segments for Automatic Vascular Model Construction},
journal={Annals of Biomedical Engineering},
year={2024},
month={Sep},
day={18},
issn={1573-9686},
doi={10.1007/s10439-024-03611-z},
url={https://doi.org/10.1007/s10439-024-03611-z},
}
```
