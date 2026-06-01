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
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python"/></a>
</p>

> **News:** SeqSeg now outputs a full SimVascular project in the `simvascular/` subdirectory — open it directly in SimVascular with automatic pathlines and contours for every segmented branch.

---

## Why SeqSeg?

SeqSeg segments vessels **sequentially**, taking steps along vessel centerlines and detecting bifurcations to grow complete vascular trees from just **1–2 seed points**. By combining local deep-learning predictions (nnU-Net) with geometric tracking, it stays robust across vessel scales — from small coronaries to large aortas.

- 🌱 **Minimal supervision** — only 1–2 seed points to initialize
- 🌿 **Robust bifurcation detection** — automatically follows every branch
- 🩻 **Multi-modal** — works with CT and MR 3D medical images
- 📏 **Scalable** — vessels from ~1mm coronaries to ~30mm aortas
- ✅ **Clinically validated** — coronary, aortic, cerebral, and pulmonary anatomies (pre-trained weights for aorta CT/MR and coronary CT)
- ⚡ **Fast** — ~2–10 min per case, Dice > 0.9 on validation, runs on CPU or GPU

<p align="center">
  <img src="https://raw.githubusercontent.com/numisveinsson/SeqSeg/main/seqseg/assets/mr_model_tracing_fast_shorter.gif" alt="SeqSeg Demo"/><br/>
  <i>Real-time demonstration: automatic segmentation of an abdominal aorta in a 3D MR scan.</i>
</p>

## What's new in 2.x

SeqSeg **2.0** refactors the package around a clearer CLI and a stable Python API. Existing batch workflows still work; legacy invocations without a subcommand (e.g. `seqseg -data_dir ...`) are rewritten to `seqseg run batch` automatically.

### Command-line interface

| Feature | Description |
| -------- | ----------- |
| **`seqseg run batch`** | Classic dataset batch tracing (preferred entry point) |
| **`seqseg run single`** | One volume + seeds: stages under `<outdir>/_seqseg_single_staging/`, then runs like batch |
| **`seqseg run plus batch`** | Global nnU-Net sweep, then SeqSeg (replaces monolithic `seqseg_plus` script flow) |
| **`seqseg init dataset`** | Scaffold `images/`, `centerlines/`, `truths/`, and template `seeds.json` |
| **`seqseg doctor`** | Check imports (SimpleITK, vtk, nnunetv2, scipy) and optional nnU-Net trainer folder |
| **`seqseg config dump` / `fingerprint`** | Inspect or diff packaged YAML configs |
| **`seqseg post global-centerline`** | Post-process segmentations into global centerlines |
| **`seqseg simvascular init`** | Create or refresh SimVascular project layout under a case directory |
| **`seqseg --version`** | Print installed package version |

### Python library API

Embed tracing in other Python code without writing SeqSeg output files:

- **`seqseg.api.run_tracing`** — pass a `sitk.Image`, seed definitions, and an nnU-Net trainer folder; get a `TracingResult` with global probability segmentation at `result.assembly.assembly`
- **`TracingOptions(disk_io=False)`** — skip VTK/MHA debug trees on disk (nnU-Net weights still load from `model_folder`)
- **`BranchSeed`**, **`branch_seed_at_point`**, **`seeds_to_potential_branches`** — simple seed formats instead of hand-built step dicts
- **`TracingContext`** / **`trace_centerline_from_context`** — lower-level control with the same in-memory image support
- Lazy re-exports from **`import seqseg`** (see `seqseg/__init__.py`)

Quick example (seeds and config known):

```python
from seqseg.api import TracingOptions, branch_seed_at_point, run_tracing

result = run_tracing(
    my_sitk_image,
    [branch_seed_at_point([x, y, z], radius)],
    "/path/to/nnUNetTrainer__nnUNetPlans__3d_fullres",
    config="global",
    options=TracingOptions(disk_io=False),
)
prob_seg = result.assembly.assembly  # sitk.Image; threshold for binary masks
```

See [High-level API (`seqseg.api`)](#high-level-api-seqsegapi) for full detail.

### Internal structure (for contributors)

- Pipeline modules: `seqseg.pipeline.classic`, `plus`, `post`, `single_trace`
- Typed config helpers: `AlgorithmConfig`, `NnUNetModelSpec` in `seqseg.config_models`
- Tracing core accepts **`sitk.Image`** or file paths for the reference volume and optional prior segmentation

### Migrating from 1.x

1. **CLI:** Prefer `seqseg run batch` (or keep legacy flags — they still work).
2. **Plus workflow:** Use `seqseg run plus batch` instead of `python -m seqseg.seqseg_plus` with the same nnU-Net path flags.
3. **Library:** Use `run_tracing` or `TracingContext` rather than calling `trace_centerline` with only file paths.
4. **Version:** `pip install -U seqseg` and check with `seqseg --version` (expects **2.0.0**).

## Quick Start

```bash
# Install
pip install seqseg

# Download pre-trained weights (see Installation docs for links)
# Run segmentation
seqseg -data_dir your_data/ -nnunet_results_path path/to/weights/ -config_name aorta_tutorial
```

**📖 New here? Follow the [step-by-step tutorial](seqseg/tutorial/tutorial.md)** with example data and detailed instructions.

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Setup, dependencies, and pre-trained model weights |
| [Usage](docs/usage.md) | Data preparation, CLI arguments, and output files |
| [Configuration](docs/configuration.md) | YAML configs and key tracking parameters |
| [Algorithm Overview](docs/algorithm.md) | Methodology, workflow, and training strategy |
| [Performance & Benchmarks](docs/benchmarks.md) | Accuracy, timing, and qualitative comparisons |
| [Research & Development](docs/development.md) | Extending SeqSeg, custom models, and tool integrations |

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
