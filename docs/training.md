# Training New Models

[← Back to README](../README.md)

SeqSeg inference needs an nnU-Net trainer folder. To train on a **new dataset**, use the optional [vascular-segment-sampler](https://pypi.org/project/vascular-segment-sampler/) integration:

```bash
pip install "seqseg[train]"
```

## 0. Set paths once

Avoid exporting `nnUNet_*` in every shell. Save defaults under `~/.seqseg/paths.yaml`:

```bash
seqseg paths init \
  --data-dir /path/to/your_project/ \
  --outdir ~/seqseg_train

# or update later:
seqseg paths set --data-dir /path/to/your_project --outdir ~/seqseg_train
seqseg paths show
```

`seqseg paths init` creates `~/nnunet_data/nnUNet_{raw,preprocessed,results}` by default (override with `--nnunet-root`).

Optional: still export into the current shell with:

```bash
eval "$(seqseg paths export)"
```

CLI flags and environment variables always override the saved file.

## 1. Prepare cases

```
your_project/
├── images/         # volumes (.nrrd, .nii.gz, …)
├── truths/         # vessel segmentations
├── centerlines/    # .vtp centerlines (required for patch sampling)
└── surfaces/       # optional; can rasterize to truths with --truth-from-surface
```

Scaffold folders with `seqseg init dataset --path your_project/`, then add centerlines and labels.

## 2. Extract patches and build an nnU-Net dataset

After `seqseg paths init` / `set`, you can omit the path flags:

```bash
seqseg train prepare \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT \
    --config-name global \
    --num-cores 4
```

Or pass them explicitly:

```bash
seqseg train prepare \
    --data-dir /path/to/your_project/ \
    --outdir /path/to/extracted/ \
    --nnunet-raw "$nnUNet_raw" \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT
```

### Resample to a target spacing

If cases should be sampled at a fixed voxel spacing, regenerate truths from `surfaces/` and resample images to match:

```bash
seqseg train prepare \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT \
    --truth-from-surface \
    --truth-regenerate \
    --truth-target-spacing 0.8 0.8 0.8 \
    --num-cores 4
```

- `--truth-from-surface` — rasterize `surfaces/` into `truths/` when needed
- `--truth-regenerate` — overwrite existing `truths/`
- `--truth-target-spacing SX SY SZ` — spacing used for the new truths (and matching image resample)

Requires `surfaces/` in the project. If you already have `truths/` and only want resampling, preprocess images/labels separately (e.g. sampler `change_img_resample`) before `seqseg train prepare`.

This wraps:

- `vascular_segment_sampler.sampling.extract_patches`
- `vascular_segment_sampler.nnunet.write_nnunet_dataset`

You can also call those APIs directly, or use the sampler CLIs `vss-sample` / `vss-to-nnunet`.

## 3. Train with nnU-Net

```bash
seqseg train nnunet --dataset-id 999 --configuration 3d_fullres --fold 0
# plan only:
seqseg train nnunet --dataset-id 999 --plan-only
# train only (after planning):
seqseg train nnunet --dataset-id 999 --skip-plan --fold all
```

Equivalent manual commands:

```bash
nnUNetv2_plan_and_preprocess -d 999
nnUNetv2_train 999 3d_fullres 0
```

## 4. Run SeqSeg with the new weights

If `data_dir`, `outdir`, and `nnunet_results` are saved via `seqseg paths`, you only need:

```bash
seqseg run batch \
    -train_dataset Dataset0999_MYDATACT \
    -fold 0 \
    -img_ext .nrrd
```

Or pass paths explicitly:

```bash
seqseg run batch \
    -train_dataset Dataset0999_MYDATACT \
    -fold 0 \
    -data_dir /path/to/inference_data/ \
    -nnunet_results_path /path/to/nnUNet_results/ \
    -img_ext .nrrd \
    -outdir results/
```

Check the environment with `seqseg doctor` (reports sampler install status and saved/effective paths).
