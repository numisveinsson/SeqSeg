#!/bin/bash
# Job name:
#SBATCH --job-name=predict_single
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio3_htc
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
#SBATCH --time=72:00:00
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

python3 test_nnunet2.py

