# Assessing the Effectiveness of Natural Language Processing (NLP) Models and Large Language Models (LLMs) for Effective Email Phishing Detection.

## Overview

### This project presents a controlled experimental comparison between:
- Decoder-only Large Language Models (LLMs)
- Encoder-only NLP Models

### For the task of email phishing detection, evaluating:
- Classification performance
- Computational efficiency
- Reliability

## HPC Environment Setup 

### 1) Load Conda
```bash
module load miniforge/24.7.1
```

### 2) Create Conda Environment
```bash
conda env create -f environment.yml
```

### 3) Activate Conda Environment
```bash
conda activate project
```


## Model Setup

### 1) Create Models Directory
```bash
mkdir models
```

### 2) Download Models From Hugging Face
Replace model_name with a real model from Hugging Face, e.g. meta-llama/Llama-3.2-3B-Instruct.
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('model_name', local_dir='/scratch/username/models/model_name')"
```

### 3) Downloading Models That Require HF Tokens
First, generate a HF Token on the Hugging Face website
Then:
```bash
export HF_TOKEN=hf_arandomstringofcharacters

python -c "from huggingface_hub import snapshot_download; snapshot_download('model_name', local_dir='/scratch/username/models/model_name', token=True)"
```


## Running Experiments

All model scripts have matching .sh or .slurm script to run it in the HPC environment, these scripts exist in the repository next to the relevant model execution script. 
However, the contents of the scripts may need to be personally modified to match your local directory structures on the HPC network.

### Baseline LLM Execution
```bash
sbatch LLM_baseline.sh
```

### Baseline NLP Model Execution
```bash
sbatch BERT_baseline_comparison.sh
sbatch RoBERTa_baseline_comparison.sh
sbatch SecBERT_baseline_comparison.sh
```

### LLM Summarisation Enhancement
```bash
sbatch LLM_summarisation.sh
```

### LLM Few-Shot Learning Enhancement
```bash
sbatch LLM_few-shot.sh
```


## Running Metrics Scripts

### Baseline LLM Metrics Execution
```bash
python metrics/LLM_baseline_metrics.py decoder_only_LLM_baseline/results/name_of_results_csv_file
```

### Baseline NLP Model Metrics Execution
```bash
python metrics/NLP_model_baseline_metrics.py encoder_only_NLP_model_baseline/results/name_of_results_csv_file
```

### LLM Summarisation Enhancement Metrics Execution
```bash
python metrics/LLM_enhancement_metrics.py decoder_only_LLM_enhancement/email_summarisation/results/name_of_results_csv_file
```

### LLM Few-Shot Learning Enhancement Metrics Execution
For Fixed Few-Shot Learning:
```bash
python metrics/LLM_enhancement_metrics.py decoder_only_LLM_enhancement/fewshot_learning/fixed_fewshot_learning/results/name_of_results_csv_file
```

For Dynamic Few-Shot Learning:
```bash
python metrics/LLM_enhancement_metrics.py decoder_only_LLM_enhancement/fewshot_learning/dynamic_fewshot_learning/results/name_of_results_csv_file
```


## Running Unit & Integration Tests

### 1) Create Venv Environment
```bash
python3 -m venv .venv
```

### 2) Activate Venv Environment
```bash
source .venv/bin/activate
```

### 3) Install pip, pytest & pandas Dependencies
```bash
python -m pip install -U pip pytest pandas
```

### 4) Run All Unit & Integration Tests
```bash
python -m pytest
```

### 5) Run Individual Tests
```bash
python -m pytest tests/test_data_scripts.py
```
```bash
python -m pytest tests/test_LLM_metrics.py
```
```bash
python -m pytest tests/test_metrics_utils.py
```
```bash
python -m pytest tests/test_NLP_model_metrics.py
```
```bash
python -m pytest tests/test_reproduce_known_stats.py
```


## Reproducibility Notes
- All Experiments use a fixed random seed of 44
- Deterministic decoding (do_sample=False) is used for LLMs
- Same dataset inside data/final_datasets is used for all experiments

## Additional Notes
- Large Models (>8B params) may require significant VRAM, typically models over 15B params didn't run on the HPC architecture
- Ensure sufficient GPU memory is available before execution by using command 'nvidia-smi'
- Some models used may require acceptance of Hugging Face terms and use of a HF Token described earlier


## This project is for academic purposes only. All datasets and models are used in accordance with their respective licenses.
