#!/bin/bash
#SBATCH --job-name=fewshot_tests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
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
SCRIPT=/users/sc23bh2/baseline_testing/trained_fewshot_tests_LLM.py
INPUT=/users/sc23bh2/baseline_testing/data/training_datasets/test_2100.csv
OUTDIR=/users/sc23bh2/baseline_testing/results/slurm_job_results/fewshot_results
SHOTS=/users/sc23bh2/baseline_testing/data/shots/shots_100.jsonl
MODELDIR=/scratch/sc23bh2/models

# Models to test
models=(
  "gemma-2-9b-it"
  "deepseek-llm-7b-chat"
  "Phi-mini-MoE-instruct"
)

# Run
for m in "${models[@]}"; do
  model_path="${MODELDIR}/${m}"

  out_csv="${OUTDIR}/results_${m}_dynamic_fewshot-20-2100.csv"

  echo "Running: ${model_path}"
  srun python "$SCRIPT" \
    --model-path "$model_path" \
    --input-csv "$INPUT" \
    --output-csv "$out_csv" \
    --shots-jsonl "$SHOTS" \
    --shots-mode "dynamic" \
    --num-phish 10 \
    --num-legit 10
  echo
done
