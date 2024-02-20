#!/bin/bash
# Job name:
#SBATCH --job-name=auto_cent
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio4_htc
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=2
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=numi@berkeley.edu
#
## Command(s) to run (example):
module load gcc
module load cuda/10.0
module load cudnn/7.5
source activate /global/scratch/users/numi/environments/seqseg2

export nnUNet_raw="/global/scratch/users/numi/nnUnet_data/nnUnet_raw"
export nnUNet_preprocessed="/global/scratch/users/numi/nnUnet_data/nnUNet_preprocessed"
export nnUNet_results="/global/scratch/users/numi/nnUnet_data/nnUNet_results"

cd /global/scratch/users/numi/SeqSeg/

python3 auto_centerline.py \
    -test_name 3d_fullres \
    -dataset  Dataset005_SEQAORTANDFEMOMR \
    -fold all \
    -img_ext .mha \
    -outdir output_v2/ \
    -scale 1 \
    -start 1 \
    -stop 2 \

# python3 auto_centerline.py \
#     -test_name 3d_fullres \
#     -dataset  Dataset010_SEQCOROASOCACT \
#     -fold all \
#     -img_ext .nrrd \
#     -outdir output_v2/ \
#     -scale 1 \
#     -start 0 \
#     -stop -1 \

#     -data_dir global/scratch/users/numi/ASOCA_test/  \

# Dataset010_SEQCOROASOCACT Dataset006_SEQAORTANDFEMOCT Dataset005_SEQAORTANDFEMOMR