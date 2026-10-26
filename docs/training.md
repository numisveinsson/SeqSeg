# Training New Models

[← Back to README](../README.md)

Inference needs an nnU-Net trainer folder. Preparing a new dataset uses [vascular-segment-sampler](https://pypi.org/project/vascular-segment-sampler/), which is installed with SeqSeg.

`seqseg train` and `seqseg paths` use hyphen flags (`--data-dir`). `seqseg run batch` uses underscore flags (`-data_dir`). Both forms are in the examples below.

## Minimum path

1. Save directories once.
2. Put images, labels, and centerlines in one project folder.
3. Build an nnU-Net dataset from local patches and preprocess it.
4. Train.
5. Point `seqseg run batch` at the new weight folder.

```bash
seqseg paths init \
  --nnunet-root ~/nnunet_data \
  --data-dir /path/to/your_project/

seqseg train prepare \
  --name MYDATA \
  --dataset-id 999 \
  --modality CT \
  --num-cores 4

seqseg train nnunet --dataset-id 999 --fold 0

seqseg run batch \
  -train_dataset Dataset999_MYDATACT \
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

`--outdir` on `seqseg paths` is where `seqseg run` writes results. `seqseg train prepare` writes extracts to `nnUNet_raw/DatasetXXX_*/_seqseg_extracted`. After conversion it deletes the subvolumes and keeps `done.txt` plus the `.txt` and `.csv` metadata in that folder.

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
    --dataset-id 999 \
    --modality CT \
    --config-name default_mm \
    --num-cores 4
```

`--config-name` here is a sampler YAML (default `default_mm`). It is a different file from the SeqSeg tracking config you pass to `seqseg run batch -config_name`. SeqSeg ships its own sampler configs in `seqseg/config/sampler/`:

- `default_mm`: millimeter data (the default). Every other config is layered on top of it.
- `default_cm`: centimeter data, such as VMR / SimVascular models.
- `test`: a quick smoke run with few samples per case.

A bare name is looked up in `./config` first, then in SeqSeg's bundled folder. You can also pass a path to your own YAML; any keys it sets override `default_mm`.

Or pass the raw directory yourself:

```bash
seqseg train prepare \
    --data-dir /path/to/your_project/ \
    --nnunet-raw "$nnUNet_raw" \
    --name MYDATA \
    --dataset-id 999 \
    --modality CT
```

Extracted patches (or whole volumes) are staged in `nnUNet_raw/DatasetXXX_*/_seqseg_extracted` and hardlinked into that dataset's `imagesTr` and `labelsTr`. Prepare then deletes the subvolume folders and leaves `done.txt`, `info_*.txt`, and `*_Sample_stats.csv`. A later `seqseg train prepare` with the same dataset skips extraction when `done.txt` already lists every case, and skips conversion when the raw images are already there. It then runs `nnUNetv2_plan_and_preprocess` for `3d_fullres`. Prepare runs it through `python -m seqseg.pipeline.nnunet_cli`, which takes the same arguments and finds the training files in one listing of `imagesTr`. nnU-Net's own lookup searches that folder once per case, which takes hours with hundreds of thousands of patches. Do not list the files under `dataset` in `dataset.json` for this: nnU-Net sends dataset.json to every preprocessing job, and a long list slows preprocessing to about one case per second. `--dataset-number` and `-dataset_number` are aliases of `--dataset-id`.

### Dataset folder name

The folder is `Dataset`, the id with at least three digits, `_`, the name, and the modality in uppercase. That is the name nnU-Net looks up from `--dataset-id`:

| `--dataset-id` | `--name` | `--modality` | Folder |
|--------------------|----------|--------------|--------|
| `5` | `MYDATA` | `CT` | `Dataset005_MYDATACT` |
| `10` | `MYDATA` | `MR` | `Dataset010_MYDATAMR` |
| `999` | `MYDATA` | `CT` | `Dataset999_MYDATACT` |

Pass that folder name as `-train_dataset` when you run SeqSeg. A comma-separated `--modality` writes one folder per modality and increments the id: `--modality CT,MR --dataset-id 999` produces `Dataset999_MYDATACT` and `Dataset1000_MYDATAMR`.

This step wraps:

- `vascular_segment_sampler.sampling.extract_patches` (default)
- `vascular_segment_sampler.sampling.gather_global_volumes` (`--global-volumes`)
- `vascular_segment_sampler.nnunet.write_nnunet_dataset`
- `nnUNetv2_plan_and_preprocess` (default configuration `3d_fullres`)

You can call the sampler steps yourself, or use the sampler commands `vss-sample`, `vss-gather-global`, and `vss-to-nnunet`.

## 3. Train

```bash
seqseg train nnunet --dataset-id 999 --fold 0
```

This only trains. Preprocessing already ran in `seqseg train prepare`. The default configuration is `3d_fullres`.

The trainer is chosen from the plans. Small patches can produce a plan that downsamples at most once (`n_stages` of 2 or fewer). nnU-Net's deep supervision makes every loss NaN on such a plan, so `seqseg train nnunet` trains it with `nnUNetTrainerNoDeepSupervision` and prints that it did. Other plans use `nnUNetTrainer`. `--trainer` overrides the choice.

The same training step, run by hand:

```bash
nnUNetv2_train 999 3d_fullres 0
```

`nnUNet_preprocessed` is the copy nnU-Net reads while training. The weight folder you need afterward is under `nnUNet_results`.

Continue a finished or interrupted run from `checkpoint_latest.pth`:

```bash
seqseg train nnunet --dataset-id 999 --fold 0 --c
```

`--continue` is the same flag.

## 4. Run SeqSeg with the new weights

With `data_dir`, `outdir`, and `nnunet_results` saved:

```bash
seqseg run batch \
    -train_dataset Dataset999_MYDATACT \
    -fold 0 \
    -img_ext .nrrd \
    -config_name global
```

`seqseg run` loads the trainer folder `<Trainer>__<Plans>__<configuration>` under that dataset whose configuration matches `-nnunet_type`, whatever the trainer is called. Keep one trainer folder per configuration: if the dataset has more than one, `seqseg run` stops and lists them, and you move the extra ones out of the dataset folder. `seqseg run single --model-folder` is the full trainer directory, so it loads whatever path you give it.

Or pass paths on the command:

```bash
seqseg run batch \
    -train_dataset Dataset999_MYDATACT \
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

`--keep-extracted` leaves the subvolumes in `nnUNet_raw/DatasetXXX_*/_seqseg_extracted`. `--outdir` stages somewhere else and is kept, including its subvolumes. `--outdir` is not deleted.

### Whole volumes

Train on each case's full image and label:

```bash
seqseg train prepare \
    --name MYDATA \
    --dataset-id 999 \
    --modality CT \
    --global-volumes
```

`--whole-volumes` is an alias for `--global-volumes`. Centerlines still decide which cases are included. `--num-cores` and `--max-samples` are ignored in this mode.

### Resample to a target spacing

Regenerate truths from `surfaces/` and resample images to match:

```bash
seqseg train prepare \
    --name MYDATA \
    --dataset-id 999 \
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

### Resume planning

Conversion writes `nnUNet_raw` and then deletes the temporary extracts. If that step finished and planning did not, plan the raw dataset in place:

```bash
seqseg train prepare --dataset-id 999 --preprocess-only --num-cores 8
```

`--name` is optional. This does not extract or convert again. Once `nnUNetPlans.json` exists, the same flag only resamples `--configuration`.

### Train a 2d model from the same subvolumes

A default prepare already stores the subvolumes in `nnUNet_raw` and writes plans that include `2d`. This follow-up only resamples those stored volumes. It does not extract patches again.

```bash
seqseg train prepare --dataset-id 999 --configuration 2d --preprocess-only --np 8
seqseg train nnunet --dataset-id 999 --configuration 2d --fold 0
```

`--name` is not required with `--preprocess-only`. A prepare that passes `--configuration 2d` without `--preprocess-only` is a first-time prepare and extracts again.

### Cleanup

`--cleanup` deletes that dataset under `nnUNet_raw` and `nnUNet_preprocessed` after training succeeds, and leaves `nnUNet_results`. A later `--c` run needs the preprocessed data, so leave it in place until you are done continuing.

```bash
seqseg train nnunet --dataset-id 999 --fold all --cleanup
```
