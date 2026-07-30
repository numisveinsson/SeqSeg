# Training New Models

[← Back to README](../README.md)

SeqSeg inference needs an nnU-Net trainer folder. To train on a **new dataset**, use the optional [vascular-segment-sampler](https://pypi.org/project/vascular-segment-sampler/) integration:

```bash
pip install "seqseg[train]"
```

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

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

seqseg train prepare \
    --data-dir /path/to/your_project/ \
    --outdir /path/to/extracted/ \
    --nnunet-raw "$nnUNet_raw" \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT \
    --config-name global \
    --num-cores 4
```

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

```bash
seqseg run batch \
    -train_dataset Dataset0999_MYDATACT \
    -fold 0 \
    -data_dir /path/to/inference_data/ \
    -nnunet_results_path "$nnUNet_results" \
    -img_ext .nrrd \
    -outdir results/
```

Check the environment with `seqseg doctor` (reports whether `vascular_segment_sampler` is installed and whether `nnUNet_*` env vars are set).
