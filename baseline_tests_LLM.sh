#!/bin/bash
#SBATCH --job-name=baseline-10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
SCRIPT=/users/sc23bh2/baseline_testing/baseline_tests_LLM.py
INPUT=/users/sc23bh2/baseline_testing/data/2200-phishing+legitimate.csv
OUTDIR=/users/sc23bh2/baseline_testing/results/slurm_job_results/results_with_type
MODELDIR=/scratch/sc23bh2/models

# Models to test
models=(
  "qwen2.5-7b-instruct"
  "llama-3.1-8b-instruct"
  "phi3-mini"
  "Llama-3.1-8B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Mistral-7B-Instruct-v0.1"
  "gemma-2-9b-it"
  "deepseek-llm-7b-chat"
  "Phi-mini-MoE-instruct"
)

# Run
for m in "${models[@]}"; do
  model_path="${MODELDIR}/${m}"

  out_csv="${OUTDIR}/results_${m}_2200-BL.csv"

  echo "Running: ${model_path}"
  srun python "$SCRIPT" \
    --model-path "$model_path" \
    --input-csv "$INPUT" \
    --output-csv "$out_csv"
  echo
done