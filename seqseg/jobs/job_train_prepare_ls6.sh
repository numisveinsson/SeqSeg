#!/bin/bash
#----------------------------------------------------
# SeqSeg: train prepare on TACC Lonestar6 (CPU, normal queue)
#
# Submit from a Lonestar6 login node:
#   sbatch seqseg/jobs/job_train_prepare_ls6.sh
#
# Edit ACCOUNT, MAIL_USER, DATA_DIR, and OUTDIR below.
# Lonestar6 charges by the node (128 cores). Raise NUM_CORES if you want.
#----------------------------------------------------

#SBATCH -J seqseg_prep
#SBATCH -o seqseg_prep.o%j
#SBATCH -e seqseg_prep.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A YOUR_ALLOCATION
#SBATCH --mail-type=all
#SBATCH --mail-user=YOUR_EMAIL@tacc.utexas.edu

set -euo pipefail

# --- edit these ---
NUM_CORES=4
SEQSEG_ENV="${SEQSEG_ENV:-/scratch/11178/numi/python-envs/seqseg}"
DATA_DIR="${DATA_DIR:-$SCRATCH/seqseg_data}"
OUTDIR="${OUTDIR:-$SCRATCH/seqseg_train}"
NNUNET_ROOT="${NNUNET_ROOT:-$SCRATCH/nnunet_data}"

module list
pwd
date
echo "host=$(hostname)  job=$SLURM_JOB_ID  cores=$NUM_CORES"

source "${SEQSEG_ENV}/bin/activate"

export OMP_NUM_THREADS="${NUM_CORES}"
export MKL_NUM_THREADS=1

which seqseg

seqseg paths init \
    --data-dir "$DATA_DIR" \
    --outdir "$OUTDIR" \
    --nnunet-root "$NNUNET_ROOT"

seqseg doctor || true

# Do not use ibrun for this (not MPI).
seqseg train prepare \
    --name MYDATA \
    --dataset-number 999 \
    --modality CT \
    --truth-from-surface \
    --truth-regenerate \
    --truth-target-spacing 0.8 0.8 0.8 \
    --num-cores "$NUM_CORES" \
    --yes

date
