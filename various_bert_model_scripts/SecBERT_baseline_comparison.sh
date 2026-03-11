#!/bin/bash
#SBATCH --job-name=secbert_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
SCRIPT=/users/sc23bh2/baseline_testing/SecBERT_baseline_comparison.py
TRAIN=/users/sc23bh2/baseline_testing/data/training_datasets/train_1540.csv
TEST=/users/sc23bh2/baseline_testing/data/training_datasets/test_660.csv
OUTDIR=/users/sc23bh2/baseline_testing/results/secbert_results
MODELDIR=/scratch/sc23bh2/models/SecBERT

# Run
srun python "$SCRIPT" \
  --model "$MODELDIR" \
  --train-csv "$TRAIN" \
  --test-csv "$TEST" \
  --output-csv "$OUTDIR/secbert_70_30_preds.csv" \
  --training-rounds 3 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --max-tokens 512