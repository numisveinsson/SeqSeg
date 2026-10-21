![test seqseg without torch build](https://github.com/numisveinsson/SeqSeg/actions/workflows/python-app.yml/badge.svg)
![test seqseg build](https://github.com/numisveinsson/SeqSeg/actions/workflows/test.yml/badge.svg)

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

## Why SeqSeg?

SeqSeg segments vessels **sequentially**, taking steps along vessel centerlines and detecting bifurcations to grow complete vascular trees from just **1–2 seed points**. By combining local deep-learning predictions (nnU-Net) with geometric tracking, it stays robust across vessel scales — from small coronaries to large aortas.

- 🌱 **Minimal supervision** — only 1–2 seed points to initialize
- 🌿 **Robust bifurcation detection** — automatically follows every branch
- 🩻 **Multi-modal** — works with CT and MR 3D medical images
- 📏 **Scalable** — vessels from ~1mm coronaries to ~30mm aortas
- ✅ **Clinically validated** — coronary, aortic, cerebral, and pulmonary anatomies (pre-trained weights for aorta CT/MR and coronary CT)
- ⚡ **Fast** — on CPU, about 2–5 minutes for an aorta, 5–15 for a coronary tree, and 10–30 for a cerebral tree. GPU is faster. Details in [Benchmarks](docs/benchmarks.md).

A finished case includes a `simvascular/` project you can open directly in SimVascular, with pathlines and contours for every segmented branch.

<p align="center">
  <img src="https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/mr_model_tracing_fast_shorter.gif" alt="SeqSeg Demo"/><br/>
  <i>Real-time demonstration: automatic segmentation of an abdominal aorta in a 3D MR scan.</i>
</p>

## Start here

- **First segmentation** — the [tutorial](seqseg/tutorial/tutorial.md) runs a sample abdominal-aorta MR scan.
- **Your own images** — [Usage](docs/usage.md) describes `images/` and `seeds.json`. The command is in [Quick Start](#quick-start).
- **Train a model** — [Training](docs/training.md).

## Quick Start

Python 3.9 or newer. 3.11 is the version used in the tutorial and conda example.

The [tutorial](seqseg/tutorial/tutorial.md) is the sample run (abdominal-aorta MR, seed points, and Windows notes). Plan on 15–20 minutes including setup. Tracing is capped, so that part finishes in a few minutes; a real case should use the defaults.

Install SeqSeg and the aorta weights below, then run `seqseg run batch` on your own data. The folder needs `images/` and `seeds.json` ([Usage](docs/usage.md)). This example is an aorta MR case. Change `-train_dataset`, `-config_name`, and `-img_ext` to match your images and weights.

```bash
pip install seqseg

# Aorta and femoral weights (MR and CT)
curl -L -o nnUNet_results.zip https://zenodo.org/records/15020477/files/nnUNet_results.zip
unzip nnUNet_results.zip
```

[Aorta and femoral weights](https://zenodo.org/records/15020477) (`nnUNet_results.zip`). [Coronary CT weights](https://zenodo.org/records/19547894) are a separate archive, `nnUNet_results_coronary.zip`.

`-nnunet_results_path` is the folder that contains `Dataset005_SEQAORTANDFEMOMR` (and the other dataset folders in that zip).

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

The tutorial uses `-config_name aorta_tutorial` for its sample scan. The trainer-folder layout is in [Installation](docs/installation.md). The full command list and flag spelling are in [Usage](docs/usage.md).

## Documentation

| Job | Guide | Description |
|-----|-------|-------------|
| Run | [Installation](docs/installation.md) | Setup, dependencies, and pre-trained model weights |
| Run | [Tutorial](seqseg/tutorial/tutorial.md) | End-to-end aorta example with sample data |
| Run | [Usage](docs/usage.md) | Data layout, seeds, CLI arguments, and output files |
| Run | [Configuration](docs/configuration.md) | YAML configs and tracking parameters |
| Train | [Training](docs/training.md) | Train nnU-Net models on a new dataset |
| Understand | [Algorithm Overview](docs/algorithm.md) | Methodology, workflow, and training strategy |
| Understand | [Performance & Benchmarks](docs/benchmarks.md) | Accuracy, timing, and qualitative comparisons |
| Integrate | [API](docs/api.md) | Call tracing from Python |
| Integrate | [Research & Development](docs/development.md) | SimVascular, 3D Slicer, and repository layout |

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
