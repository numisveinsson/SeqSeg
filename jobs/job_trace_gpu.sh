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
#SBATCH --time=12:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=numi@berkeley.edu
#
## Command(s) to run (example):

source activate /global/scratch/users/numi/environments/seqseg2

export nnUNet_raw="/global/scratch/users/numi/nnUnet_data/nnUnet_raw"
export nnUNet_preprocessed="/global/scratch/users/numi/nnUnet_data/nnUNet_preprocessed"
export nnUNet_results="/global/scratch/users/numi/nnUnet_data/nnUNet_results"

cd /global/scratch/users/numi/SeqSeg/

python3 auto_centerline.py \
    -test_name 3d_fullres \
    -train_dataset Dataset024_SEQAORTAS050MR \
    -fold 0 \
    -img_ext .mha \
    -outdir output_datasizestudy/mr/050/ \
    -scale 1 \
    -start 0 \
    -stop -1 \
    -max_n_steps 500 \
    -max_n_branches 50 \
    -unit cm \
    -config_name global_aorta \
    # -data_dir /global/scratch/users/numi/MICCAI_AVT_Data/  \

# Dataset010_SEQCOROASOCACT Dataset006_SEQAORTANDFEMOCT Dataset005_SEQAORTANDFEMOMR Dataset016_SEQPULMPARSECT
# Dataset018_SEQAORTASONEMR Dataset017_SEQAORTASONECT