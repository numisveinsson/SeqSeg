# Python API

[← Back to README](../README.md)

Use this page to call SeqSeg from Python. Command-line use is documented in [Usage](usage.md).

Weights still load from an nnU-Net trainer folder on disk (`…/nnUNetTrainer__nnUNetPlans__3d_fullres`). Seed coordinates and the seed radius use the same unit as `TracingOptions.unit` (default `cm`). Lengths inside a YAML config stay in millimeters; see [Configuration](configuration.md).

`import seqseg` loads these names lazily. Importing from `seqseg.api` is the same surface for tracing.

## Trace one image

`run_tracing` takes a `sitk.Image`, seeds, and a trainer folder. It returns a `TracingResult`. The accumulated probability volume is `result.assembly.assembly` (a `sitk.Image`). Threshold that image for a binary mask.

```python
import SimpleITK as sitk
from seqseg.api import TracingOptions, branch_seed_at_point, run_tracing

image = sitk.ReadImage("volume.mha")
result = run_tracing(
    image,
    [branch_seed_at_point([-2.07, -2.20, 13.43], radius=1.1)],
    "/path/to/nnUNet_results/Dataset005_SEQAORTANDFEMOMR/nnUNetTrainer__nnUNetPlans__3d_fullres",
    config="global_aorta",
    options=TracingOptions(disk_io=False, unit="cm"),
)
prob = result.assembly.assembly
```

`config` is a packaged YAML name (`"global"`, `"global_aorta"`, `"global_coro"`, …), an `AlgorithmConfig`, or a plain mapping of the same keys.

`disk_io=False` skips writing VTK and MHA debug trees. Set `output_folder` when `disk_io=True`.

### `run_tracing` arguments

| Argument | Meaning |
|----------|---------|
| `image` | Reference volume |
| `seeds` | See Seeds below |
| `model_folder` | nnU-Net trainer folder |
| `case` | Label used in logs and in any files written when `disk_io` is true. Default `seqseg_case` |
| `config` | Packaged config name, `AlgorithmConfig`, or mapping. Default `"global"` |
| `options` | `TracingOptions`. Defaults match a library call |
| `output_folder` | Output root when `disk_io` is true |

### `TracingOptions`

| Field | Default | Meaning |
|-------|---------|---------|
| `max_n_steps` | `1000` | Maximum tracking steps |
| `max_n_branches` | `100` | Maximum branches |
| `max_n_steps_per_branch` | `100` | Maximum steps on one branch |
| `write_samples` | `False` | Write intermediate patches |
| `disk_io` | `True` | Write debug trees. Set `False` to keep results in memory |
| `simvascular` | `False` | Write a SimVascular project when writing to disk |
| `unit` | `"cm"` | Unit of seed coordinates and seed radius |
| `scale` | `1.0` | Multiplies voxel spacing passed to nnU-Net. `0.1` converts mm spacing to cm |
| `fold` | `"all"` | nnU-Net fold |
| `force_cpu` | `False` | Run nnU-Net on CPU |
| `seg_file` | `None` | Optional prior segmentation, path or `sitk.Image` |
| `start_seg` | `None` | Optional initial mask. A path is loaded and resampled onto `image`; the SeqSeg result is merged into it |

### `TracingResult`

| Field | Meaning |
|-------|---------|
| `assembly` | Object whose `.assembly` attribute is the probability `sitk.Image` |
| `centerlines` | Local centerlines from the trace |
| `surfaces` | Local surfaces |
| `points` | Tracking points |
| `inside_pts` | Points classified inside the vessel |
| `vessel_tree` | Assembled tree |
| `n_steps_taken` | Steps actually taken |

## Seeds

Each seed is a start point, a point further along the vessel (the direction), and a lumen radius in the same unit as `options.unit`.

`BranchSeed(old_point, new_point, radius)` is that triple. `old_point` is the start, `new_point` is the direction point.

`branch_seed_at_point(point, radius, tangent=None, step=1.0)` builds a `BranchSeed` from one point. The start is `point - step * tangent`. The default tangent is `(0, 0, 1)`.

`seeds_to_potential_branches` accepts a sequence of any of:

- `BranchSeed`
- a mapping with keys `old_point`, `new_point`, `radius`
- a length-3 sequence `(old_point, new_point, radius)`
- a length-2 sequence `(point, radius)`, which uses `branch_seed_at_point`

`run_tracing` calls `seeds_to_potential_branches` for you.

Two points from the aorta tutorial (`unit="cm"`, radius `1.1`):

```python
from seqseg.api import BranchSeed, run_tracing

seeds = [BranchSeed(
    old_point=[-2.07367, -2.1973, 13.4288],
    new_point=[-1.17086, -1.33526, 12.2407],
    radius=1.1,
)]
```

## Lower-level tracing

`TracingContext` plus `trace_centerline_from_context` is the same trace with the step dicts already built. Prefer `run_tracing` when you have points and a radius.

```python
import numpy as np
import SimpleITK as sitk
from seqseg import AlgorithmConfig
from seqseg.api import seeds_to_potential_branches, branch_seed_at_point
from seqseg.modules.tracing import TracingContext, trace_centerline_from_context

image = sitk.ReadImage("volume.mha")
ctx = TracingContext(
    output_folder="",
    image_file=image,
    case="api",
    model_folder="/path/to/nnUNetTrainer__nnUNetPlans__3d_fullres",
    fold="all",
    potential_branches=seeds_to_potential_branches([
        branch_seed_at_point([-2.07, -2.20, 13.43], 1.1),
    ]),
    max_step_size=1000,
    max_n_branches=100,
    max_n_steps_per_branch=100,
    global_config=AlgorithmConfig.from_name("global_aorta"),
    unit="cm",
    disk_io=False,
    write_samples=False,
)
result = trace_centerline_from_context(ctx)
```

`image_file` may be a filesystem path or a `sitk.Image`. `max_step_size` on the context is `max_n_steps` on `TracingOptions`.

`AlgorithmConfig.from_name("global_aorta")` loads `seqseg/config/global_aorta.yaml`. `load_yaml_config` returns the same mapping as a plain dict. `NnUNetModelSpec` builds the trainer-folder path from a dataset name, nnU-Net configuration, and results directory:

```python
from seqseg import NnUNetModelSpec

folder = NnUNetModelSpec(
    train_dataset="Dataset005_SEQAORTANDFEMOMR",
    results_path="/path/to/nnUNet_results",
).model_folder()
```

## Batch and post-processing helpers

These are the functions the CLI calls. The commands in [Usage](usage.md) are the supported way to run them.

| Name | CLI |
|------|-----|
| `run_classic_batch` | `seqseg run batch` |
| `run_plus_batch` | `seqseg run plus batch` |
| `bootstrap_simvascular_project` | `seqseg simvascular init` |
| `bootstrap_simvascular_project_batch` | `seqseg simvascular init-batch` |
| `run_global_centerline_single` | `seqseg post global-centerline single` |
| `run_global_centerline_batch` | `seqseg post global-centerline batch` |

## Migrating library calls

Call `run_tracing` or `trace_centerline_from_context` with a `sitk.Image`. `trace_centerline` still accepts a volume path when you already have step dicts and want the lower-level function.
