#!/bin/bash
#SBATCH --job-name=logical-contradiction-emails
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/users/sc23bh2/baseline_testing/logs/%x-%j.out
#SBATCH --error=/users/sc23bh2/baseline_testing/logs/%x-%j.err

set -euo pipefail

# Environment / dependencies
module load miniforge/24.7.1
conda activate project

# Cache locations on scratch
export HF_HOME=/scratch/sc23bh2/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export CONDA_PKGS_DIRS=/scratch/sc23bh2/conda/pkgs
export CONDA_ENVS_PATH=/scratch/sc23bh2/conda/envs
export PIP_CACHE_DIR=/scratch/sc23bh2/pip-cache

# Paths
SCRIPT=/users/sc23bh2/baseline_testing/email_manipulation/email_manipulator.py
INDIR=/users/sc23bh2/baseline_testing/data/2200-phishing+legitimate.csv
OUTDIR=/users/sc23bh2/baseline_testing/email_manipulation/logical_contradiction_emails_2200.csv
MODELDIR=/scratch/sc23bh2/models/gemma-2-9b-it

# Run
srun python "$SCRIPT" \
  --model "$MODELDIR" \
  --input-csv "$INDIR" \
  --output-csv "$OUTDIR" \
  --strategy "logical_contradiction"
