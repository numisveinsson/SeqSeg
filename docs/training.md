# Training New Models

[← Back to README](../README.md)

Inference needs an nnU-Net trainer folder. To train one on a new dataset, install the optional [vascular-segment-sampler](https://pypi.org/project/vascular-segment-sampler/) extra:

```bash
pip install "seqseg[train]"
```

`seqseg train` and `seqseg paths` use hyphen flags (`--data-dir`). `seqseg run batch` uses underscore flags (`-data_dir`). Both forms are in the examples below.

## Minimum path

1. Save directories once.
2. Put images, labels, and centerlines in one project folder.
3. Build an nnU-Net dataset from local patches.
4. Train.
5. Point `seqseg run batch` at the new weight folder.

```bash
seqseg paths init \
  --nnunet-root ~/nnunet_data \
  --data-dir /path/to/your_project/

seqseg train prepare \
  --name MYDATA \
  --dataset-number 999 \
  --modality CT \
  --num-cores 4

seqseg train nnunet --dataset-id 999 --configuration 3d_fullres --fold 0

seqseg run batch \
  -train_dataset Dataset0999_MYDATACT \
  -fold 0 \
  -img_ext .nrrd
```

The last command uses `data_dir`, `outdir`, and `nnunet_results` from `seqseg paths`. Details for each step follow.

## 0. Set paths once

Training uses one nnU-Net root. Models land in `nnUNet_results` under that root. Save the root in `~/.seqseg/paths.yaml` so later commands can omit the path flags:

```bash
seqseg paths init \
  --nnunet-root ~/nnunet_data \
  --data-dir /path/to/your_project/

seqseg paths show
```

`seqseg paths init` creates `~/nnunet_data/nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`. Another root is `--nnunet-root`.

Change them later with `seqseg paths set`. That command requires `--nnunet-root` and rewrites the three nnU-Net directories under that root:

```bash
seqseg paths set \
  --nnunet-root ~/nnunet_data \
  --data-dir /path/to/your_project
```

`--outdir` on `seqseg paths` is where `seqseg run` writes results. `seqseg train prepare` writes temporary extracts to `<nnunet-root>/_seqseg_extracted` and removes that folder after linking them into `nnUNet_raw`.

To export the saved paths into the current shell:

```bash
eval "$(seqseg paths export)"
```

CLI flags and environment variables override the saved file.

## 1. Prepare cases

```
your_project/
├── images/         # volumes (.nrrd, .nii.gz, …)
├── truths/         # vessel segmentations
├── centerlines/    # .vtp centerlines (required to list cases, including whole-volume mode)
└── surfaces/       # optional; rasterize to truths with --truth-from-surface
```

Scaffold the folders with `seqseg init dataset --path your_project/`, then add images, centerlines, and labels.

## 2. Build the nnU-Net dataset

After `seqseg paths init` or `seqseg paths set`, omit the path flags:

```bash
seqseg train prepare \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT \
    --config-name global \
    --num-cores 4
```

`--config-name` here is the sampler YAML from vascular-segment-sampler (default `global`). It is a different file from the SeqSeg tracking config you pass to `seqseg run batch -config_name`.

Or pass the raw directory yourself:

```bash
seqseg train prepare \
    --data-dir /path/to/your_project/ \
    --nnunet-raw "$nnUNet_raw" \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT
```

Extracted patches (or whole volumes) are staged in `<parent of nnUNet_raw>/_seqseg_extracted`, hardlinked into `nnUNet_raw/DatasetXXX_*`, then deleted.

### Dataset folder name

Ids below 10 are written as two digits (`5` → `05`). The folder is `Dataset0`, then that id, then `_`, the name, and the modality in uppercase:

| `--dataset-number` | `--name` | `--modality` | Folder |
|--------------------|----------|--------------|--------|
| `5` | `MYDATA` | `CT` | `Dataset005_MYDATACT` |
| `10` | `MYDATA` | `MR` | `Dataset010_MYDATAMR` |
| `999` | `MYDATA` | `CT` | `Dataset0999_MYDATACT` |

Pass that folder name as `-train_dataset` when you run SeqSeg. A comma-separated `--modality` writes one folder per modality and increments the id: `--modality CT,MR --dataset-number 999` produces `Dataset0999_MYDATACT` and `Dataset01000_MYDATAMR`.

This step wraps:

- `vascular_segment_sampler.sampling.extract_patches` (default)
- `vascular_segment_sampler.sampling.gather_global_volumes` (`--global-volumes`)
- `vascular_segment_sampler.nnunet.write_nnunet_dataset`

You can call those yourself, or use the sampler commands `vss-sample`, `vss-gather-global`, and `vss-to-nnunet`.

## 3. Train

```bash
seqseg train nnunet --dataset-id 999 --configuration 3d_fullres --fold 0
```

`--configuration` is passed to nnU-Net preprocessing as `-c` (default `3d_fullres`) and to training.

The same thing, run by hand:

```bash
nnUNetv2_plan_and_preprocess -d 999 -c 3d_fullres
nnUNetv2_train 999 3d_fullres 0
```

`nnUNet_preprocessed` is a second copy nnU-Net uses while training. The weight folder you need afterward is under `nnUNet_results`.

## 4. Run SeqSeg with the new weights

With `data_dir`, `outdir`, and `nnunet_results` saved:

```bash
seqseg run batch \
    -train_dataset Dataset0999_MYDATACT \
    -fold 0 \
    -img_ext .nrrd \
    -config_name global
```

Or pass paths on the command:

```bash
seqseg run batch \
    -train_dataset Dataset0999_MYDATACT \
    -fold 0 \
    -data_dir /path/to/inference_data/ \
    -nnunet_results_path /path/to/nnUNet_results/ \
    -img_ext .nrrd \
    -outdir results/
```

`-config_name global` is a starting tracking config. Switch to `global_aorta`, `global_coro`, or your own file when the vessel size matches one of those. See [Configuration](configuration.md).

`seqseg doctor` reports whether the sampler is installed and which paths are saved.

## Options

Skip these on a first training run.

### Keep the extracted patches

`--keep-extracted` leaves `<nnunet-root>/_seqseg_extracted` in place. `--outdir` stages somewhere else and is kept (use it to resume with `--skip-sample`). `--outdir` is not deleted.

### Whole volumes

Train on each case's full image and label:

```bash
seqseg train prepare \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT \
    --global-volumes \
    --yes
```

`--whole-volumes` is an alias for `--global-volumes`. Centerlines still decide which cases are included. `--num-cores` and `--max-samples` are ignored in this mode.

### Resample to a target spacing

Regenerate truths from `surfaces/` and resample images to match:

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

- `--truth-from-surface` rasterizes `surfaces/` into `truths/` when needed
- `--truth-regenerate` overwrites existing `truths/`
- `--truth-target-spacing SX SY SZ` is the spacing of the new truths and of the resampled images

This requires `surfaces/` in the project. If `truths/` already exist and you only want resampling, resample images and labels before `seqseg train prepare` (the sampler command `change_img_resample` does this). These flags also apply with `--global-volumes`.

### Plan and train separately

```bash
seqseg train nnunet --dataset-id 999 --configuration 3d_fullres --plan-only
seqseg train nnunet --dataset-id 999 --skip-plan --fold all
```

`--cleanup` deletes that dataset under `nnUNet_raw` and `nnUNet_preprocessed` after training succeeds, and leaves `nnUNet_results`. It is ignored with `--plan-only`, so later folds can reuse the preprocessed data.

```bash
seqseg train nnunet --dataset-id 999 --skip-plan --fold all --cleanup
```
