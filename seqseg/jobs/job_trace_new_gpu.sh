#!/bin/bash
# Job name:
#SBATCH --job-name=seqseg
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# QoS:
#SBATCH --qos=gtx2080_gpu3_normal
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
# Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=numi@berkeley.edu
#
## Command(s) to run (example):

module load python/3.11.6-gcc-11.4.0
module load ml/pytorch/2.3.1-py3.11.7
# pip install nnunetv2
export PATH="/global/home/users/numi/.local/bin:$PATH"

export nnUNet_raw="/global/scratch/users/numi/nnUnet_data/nnUnet_raw"
export nnUNet_preprocessed="/global/scratch/users/numi/nnUnet_data/nnUNet_preprocessed"
export nnUNet_results="/global/scratch/users/numi/nnUnet_data/nnUNet_results"

cd /global/scratch/users/numi/SeqSeg/

python3 seqseg.py \
    -test_name 3d_fullres \
    -train_dataset Dataset048_SEQAORTAVMRGALACT \
    -fold 0 \
    -img_ext .mha \
    -outdir output_gala_aaas_more_seeds/ \
    -scale 1 \
    -start 0 \
    -stop -1 \
    -max_n_steps 500 \
    -max_n_branches 30 \
    -unit cm \
    -config_name global_aorta \
    -pt_centerline 1 \
    -num_seeds_centerline 15 \

# Dataset010_SEQCOROASOCACT Dataset006_SEQAORTANDFEMOCT Dataset005_SEQAORTANDFEMOMR Dataset016_SEQPULMPARSECT
# Dataset018_SEQAORTASONEMR Dataset017_SEQAORTASONECT