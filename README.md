# TRUST-RAG V2: Evidence-Grounded Trust-Aware RAG for Biomedical QA

## Overview

TRUST-RAG is a modular research framework that builds a biomedical RAG system
and independently measures how well generated answers are supported by
retrieved evidence — separating **answer correctness** from **answer
trustworthiness**.

## Project Structure

```
TRUST-RAG/
├── config/               # YAML hyperparameters & experiment configs
├── data/                 # Raw & processed datasets
├── src/                  # Core library modules
├── notebooks/            # Analysis & visualization notebooks
├── experiments/          # Runnable experiment scripts
└── results/              # All experiment outputs (JSONL, JSON, PNG)
```

## Quick Start

### Option A: Google Colab (Recommended for T4 GPU)
Open [`notebooks/TRUST_RAG_Colab_Pipeline.ipynb`](file:///Users/kartikeybhatt/.gemini/antigravity-ide/scratch/TRUST-RAG/notebooks/TRUST_RAG_Colab_Pipeline.ipynb) directly in Google Colab. The notebook walks through all stages sequentially:
1. Environment & GPU check (T4 GPU verified)
2. Evidence indexing with MedCPT & BM25
3. Retrieval benchmarking (BM25 vs MedCPT vs Hybrid RRF)
4. Generation (MedGemma 1.5 4B-IT fp16, then Qwen 2.5 7B 4-bit)
5. Decision extraction & DeBERTa NLI claim verification
6. Calibrated trust model training (LogReg + Platt scaling)
7. Evaluation & Decorrelation analysis
8. 6-Category error taxonomy classification
9. Systematic ablation study & 300-DPI publication figures

### Option B: Local / CLI Execution
```bash
pip install -r requirements.txt

# 1. Prepare data splits
python -m src.data_loader

# 2. Build indices
python -m src.indexer

# 3. Run full pipeline end-to-end
python experiments/run_full_pipeline.py --model medgemma --split val
python experiments/run_full_pipeline.py --model qwen --split val

# 4. Run systematic ablation study
python experiments/run_ablation.py --split val

# 5. Generate publication figures & LaTeX/Markdown tables
python experiments/generate_figures.py --split val
```

## Hardware

Designed for **Google Colab with Tesla T4** (15 GB VRAM, 12.7 GB RAM).
Sequential model loading ensures no memory conflicts.

## Key Components

| Module | Purpose |
|--------|---------|
| `retriever.py` | Hybrid BM25 + MedCPT retrieval with RRF fusion |
| `generator.py` | Unified LLM interface (Qwen 2.5 7B / MedGemma 4B) |
| `nli_verifier.py` | DeBERTa NLI claim-level evidence verification |
| `trust_model.py` | Calibrated logistic regression trust estimator |
| `evaluation.py` | Correctness + trust metrics + calibration analysis |

## Citation

If you use this code, please cite the TRUST-RAG project.
