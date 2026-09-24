"""
Script to generate TRUST_RAG_Colab_Pipeline.ipynb with self-healing Colab support.
"""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # ── CELL 1: Header ──────────────────────────────────────────────────────────
    add_md("""# TRUST-RAG V2: Evidence-Grounded Trust-Aware Biomedical Question Answering
### End-to-End Google Colab Research Pipeline (T4 GPU Optimized)

---

**Research Objectives:**
1. **Disentangled Evaluation**: Measure **answer correctness** (vs PubMedQA expert gold labels) separately from **answer trustworthiness** (evidence grounding & hallucination detection).
2. **Hybrid Evidence Retrieval**: BM25 Okapi + MedCPT Dense Embedding with Reciprocal Rank Fusion ($k=60$) and PMID-level document grouping.
3. **Multi-Model Benchmark**: Compare biomedical-specialized **MedGemma 1.5 4B IT** (fp16) against **Qwen 2.5 7B Instruct** (4-bit NF4).
4. **Claim-Level NLI Verification**: Decompose reasoning into atomic claims and verify against retrieved evidence using `cross-encoder/nli-deberta-v3-base`.
5. **Calibrated Trust Estimation**: 9-feature vector mapped via Platt-calibrated L2 Logistic Regression into empirical probability of correctness.
6. **Trust-Correctness Decorrelation**: 2×2 Contingency analysis isolating *Dangerous* (High Trust + Wrong) and *Lucky* (Low Trust + Correct) failures.
7. **6-Category Error Taxonomy**: Root-cause failure analysis (E1 Retrieval Failure through E6 Misinterpretation).

---
> **Colab T4 Memory Safety**: All neural models are loaded sequentially onto the GPU and memory is cleared with `clear_gpu_memory()` after each step to prevent CUDA Out-Of-Memory (OOM) errors.""")

    # ── CELL 2: GPU Check ───────────────────────────────────────────────────────
    add_md("## Step 0: Hardware & Environment Verification\nChecks that a GPU (preferably NVIDIA T4 or better) is attached to the Colab runtime.")
    add_code("""# Verify GPU
!nvidia-smi

import torch
print(f"\\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: No GPU detected! Go to Runtime -> Change runtime type -> Hardware accelerator: T4 GPU")
""")

    # ── CELL 3: Configuration & Demo Mode ───────────────────────────────────────
    add_md("### Experiment Configuration (Demo Mode vs Full Run)\nChoose whether to run a quick 3-minute validation run or the full 200-sample benchmark.")
    add_code("""import os

# ── RUNTIME SETTINGS ──────────────────────────────────────────────────────────
# Set DEMO_MODE = True to test all 12 pipeline stages on 10 samples in ~2-3 minutes.
# Set DEMO_MODE = False to run the full benchmark on all 200 validation queries.
DEMO_MODE = True
MAX_SAMPLES = 10 if DEMO_MODE else None

# Hugging Face Token (Optional):
# Only required if you wish to download google/medgemma-1.5-4b-it (which is gated).
# If left empty, the pipeline will automatically use Qwen 2.5 7B / 3B without needing any login.
HF_TOKEN = ""

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("✓ HF_TOKEN registered in environment")
else:
    print("ℹ️ Running in open-access mode (no HF token required for Qwen models)")

print(f"✓ Configuration: DEMO_MODE={DEMO_MODE}, MAX_SAMPLES={MAX_SAMPLES}")
""")

    # ── CELL 4: Dependency Installation ─────────────────────────────────────────
    add_md("### Install Dependencies\nInstalls all required scientific and deep learning packages.")
    add_code("""# Install dependencies cleanly
!pip install -q --upgrade pip
!pip install -q torch transformers accelerate bitsandbytes sentencepiece rank-bm25 faiss-cpu scikit-learn pyyaml pandas numpy matplotlib seaborn
!pip install -q spacy
!python -m spacy download en_core_web_sm -q

print("✓ All libraries installed successfully!")
""")

    # ── CELL 5: Project Setup & Auto-Discovery ──────────────────────────────────
    add_md("### Setup Project Directory & Unpack Files\nLocates or extracts the TRUST-RAG repository and adds `src` to `sys.path`.")
    add_code("""import os
import sys
from pathlib import Path

# Helper function to find project root
def locate_and_set_root():
    candidates = [
        Path.cwd(),
        Path("/content/TRUST-RAG"),
        Path("/content"),
        Path.cwd() / "TRUST-RAG",
        Path("/content/drive/MyDrive/TRUST-RAG"),
    ]
    for c in candidates:
        if (c / "src").is_dir() and (c / "src" / "config.py").is_file():
            os.chdir(str(c))
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            print(f"✓ Found TRUST-RAG repository at: {c}")
            print(f"✓ Working directory set to: {os.getcwd()}")
            print(f"✓ Added to sys.path: {c}")
            return True

    # Check if TRUST-RAG.zip exists in /content or current directory
    for z in [Path("/content/TRUST-RAG.zip"), Path("TRUST-RAG.zip"), Path("/content/trust_rag.zip")]:
        if z.is_file():
            print(f"✓ Detected {z}! Extracting into /content/TRUST-RAG ...")
            os.makedirs("/content/TRUST-RAG", exist_ok=True)
            os.system(f"unzip -q -o {z} -d /content/TRUST-RAG")
            return locate_and_set_root()

    print("⚠️ Project root not found yet.")
    print("Please upload TRUST-RAG.zip to the Colab files panel on the left and re-run this cell!")
    return False

locate_and_set_root()
!ls -la
""")

    # ── CELL 6: Step 1 Data Loader ──────────────────────────────────────────────
    add_md("## Step 1: Data Loading & Dataset Verification\nLoads PubMedQA dataset splits (`train`: 600, `val`: 200, `test`: 200).")
    add_code("""import sys, os
from pathlib import Path

# Ensure project root in sys.path
for _p in [Path.cwd(), Path("/content/TRUST-RAG"), Path("/content")]:
    if (_p / "src").is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config import get_default_config, resolve_path
from src.data_loader import run as run_data_loader
from src.utils import load_json, load_jsonl

cfg = get_default_config()
val_jsonl = resolve_path(cfg["data"]["processed_dir"]) / "val.jsonl"
raw_csv = resolve_path(cfg["data"]["raw_path"])

# If splits do not exist, run data loader
if not val_jsonl.exists():
    if raw_csv.exists():
        print(f"Creating splits from {raw_csv}...")
        run_data_loader(cfg)
    else:
        raise FileNotFoundError(f"Neither {val_jsonl} nor {raw_csv} was found. Ensure TRUST-RAG files are extracted!")

# Load dataset statistics
stats_path = resolve_path(cfg["data"]["processed_dir"]) / "dataset_stats.json"
if stats_path.exists():
    stats = load_json(stats_path)
    print("Dataset Summary:")
    print(f"  Total records: {stats.get('n_records')}")
    print(f"  Class breakdown: {stats.get('class_distribution')}")
    print(f"  Class percentages: {stats.get('class_percentages')}")

# Load validation records
all_val_records = load_jsonl(val_jsonl)
val_records = all_val_records[:MAX_SAMPLES] if MAX_SAMPLES else all_val_records
print(f"\\n✓ Loaded {len(val_records)} validation records (Total in split: {len(all_val_records)})")
print(f"Sample Question: {val_records[0]['question']}")
print(f"Sample Gold Decision: {val_records[0]['final_decision']}")
""")

    # ── CELL 7: Step 2 Evidence Indexing ────────────────────────────────────────
    add_md("## Step 2: Evidence Indexing (MedCPT + BM25)\nEncodes all PubMed passages with `ncbi/MedCPT-Article-Encoder` into a FAISS Inner-Product index and builds a BM25 Okapi index.")
    add_code("""from src.indexer import run as run_indexer
from src.config import get_default_config, resolve_path
from src.utils import clear_gpu_memory

cfg = get_default_config()
faiss_path = resolve_path(cfg["retrieval"]["faiss_index_path"])
bm25_path = resolve_path(cfg["retrieval"]["bm25_index_path"])

if not (faiss_path.exists() and bm25_path.exists()):
    print("Building BM25 and MedCPT FAISS indices on GPU...")
    index_res = run_indexer(cfg)
    print("Indexing complete:", index_res)
else:
    print(f"✓ Indices already exist at {faiss_path} and {bm25_path}")

clear_gpu_memory()
""")

    # ── CELL 8: Step 3 Retrieval Evaluation ─────────────────────────────────────
    add_md("## Step 3: Hybrid Retrieval & Comparison\nEvaluates BM25, MedCPT Dense, and Hybrid RRF fusion on the validation queries and caches evidence blocks.")
    add_code("""import pandas as pd
from src.retriever import HybridRetriever
from src.retrieval_eval import evaluate_retrieval
from src.utils import clear_gpu_memory, load_jsonl, save_jsonl

retriever = HybridRetriever.from_config(cfg)

# Benchmark retrieval modes
retrieval_comparison = {}
for mode in ["bm25", "dense", "hybrid"]:
    res = evaluate_retrieval(retriever, val_records, ks=[1, 3, 5], top_k=5, mode=mode)
    retrieval_comparison[mode.upper()] = res["aggregate"]

df_ret = pd.DataFrame(retrieval_comparison).T[["recall_at_1", "recall_at_3", "recall_at_5", "mrr", "n_found"]]
print("Retrieval Performance on Validation Queries:")
display(df_ret)

# Cache retrieved evidence for generation
evidence_cache = {}
retrieval_records = []
for rec in val_records:
    pmid = str(rec["pmid"])
    blocks = retriever.retrieve(rec["question"], top_k=cfg["retrieval"]["evidence_top_k"], mode="hybrid")
    evidence_cache[pmid] = blocks
    retrieval_records.append({
        "question_pmid": pmid,
        "evidence_pmids": [b.pmid for b in blocks],
        "evidence_scores": [round(b.score, 6) for b in blocks],
        "evidence_texts": [b.text for b in blocks],
        "n_chunks": [b.n_chunks for b in blocks],
    })

ret_cache_path = resolve_path(cfg["results"]["base_dir"]) / "retrieval" / "val_retrieval.jsonl"
save_jsonl(retrieval_records, ret_cache_path)
print(f"✓ Cached {len(retrieval_records)} retrieved evidence blocks -> {ret_cache_path}")

# Free query encoder
retriever.unload_query_encoder()
del retriever
clear_gpu_memory()
""")

    # ── CELL 9: Step 4 Generation Model A (MedGemma) ───────────────────────────
    add_md("## Step 4: Generation — Model A: MedGemma 1.5 4B-IT\nGenerates evidence-grounded answers with reasoning using `google/medgemma-1.5-4b-it` (fp16).\n*(Note: If Gemma gated access is not configured, the cell will automatically fall back to Qwen 2.5 3B Instruct)*.")
    add_code("""from src.generator import BiomedicalGenerator
from src.utils import clear_gpu_memory

gen_output_medgemma = resolve_path(cfg["results"]["base_dir"]) / "generation" / "medgemma_val_outputs.jsonl"

try:
    print("Loading Model A (MedGemma 1.5 4B IT fp16)...")
    medgemma_gen = BiomedicalGenerator.from_config(cfg, model_key="medgemma")
    medgemma_gen.load_model()
except Exception as e:
    print(f"\\nNotice: Could not load gated MedGemma model: {e}")
    print("Switching Model A to open-access alternative: Qwen/Qwen2.5-3B-Instruct...")
    cfg["generation"]["models"]["medgemma"]["name"] = "Qwen/Qwen2.5-3B-Instruct"
    medgemma_gen = BiomedicalGenerator.from_config(cfg, model_key="medgemma")
    medgemma_gen.load_model()

print(f"Generating answers for {len(val_records)} validation questions...")
medgemma_results = medgemma_gen.generate_batch(
    records=val_records,
    evidence_map=evidence_cache,
    output_path=gen_output_medgemma,
    resume=True,
)

print(f"✓ Model A generation complete! Output saved to: {gen_output_medgemma}")
medgemma_gen.unload_model()
del medgemma_gen
clear_gpu_memory()
""")

    # ── CELL 10: Step 5 Generation Model B (Qwen 2.5 7B) ────────────────────────
    add_md("## Step 5: Generation — Model B: Qwen 2.5 7B-Instruct\nGenerates answers using `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 quantization (~5.5GB VRAM).")
    add_code("""from src.generator import BiomedicalGenerator
from src.utils import clear_gpu_memory

gen_output_qwen = resolve_path(cfg["results"]["base_dir"]) / "generation" / "qwen_val_outputs.jsonl"

print("Loading Model B (Qwen 2.5 7B Instruct 4-bit NF4)...")
qwen_gen = BiomedicalGenerator.from_config(cfg, model_key="qwen")
qwen_gen.load_model()

print(f"Generating answers for {len(val_records)} validation questions...")
qwen_results = qwen_gen.generate_batch(
    records=val_records,
    evidence_map=evidence_cache,
    output_path=gen_output_qwen,
    resume=True,
)

print(f"✓ Model B generation complete! Output saved to: {gen_output_qwen}")
qwen_gen.unload_model()
del qwen_gen
clear_gpu_memory()
""")

    # ── CELL 11: Step 6 Decision Extraction & NLI Verification ──────────────────
    add_md("## Step 6: Decision Extraction & NLI Verification\n1. Extracts decisions (`yes`/`no`/`maybe`) using rule-based parser + zero-shot NLI.\n2. Decomposes answers into atomic claims.\n3. Verifies claims against retrieved evidence using `cross-encoder/nli-deberta-v3-base`.")
    add_code("""from src.decision_extractor import DecisionExtractor
from src.claim_extractor import ClaimExtractor
from src.nli_verifier import NLIVerifier
from src.utils import clear_gpu_memory, load_jsonl, save_jsonl

extractor = DecisionExtractor.from_config(cfg)
claim_extractor = ClaimExtractor.from_config(cfg)
nli_verifier = NLIVerifier.from_config(cfg)

model_keys = ["medgemma", "qwen"]
processed_data = {}

for m in model_keys:
    print(f"\\n--- Verification Pipeline for {m.upper()} ---")
    gen_path = resolve_path(cfg["results"]["base_dir"]) / "generation" / f"{m}_val_outputs.jsonl"
    gen_outputs = load_jsonl(gen_path)
    
    # 1. Decision extraction
    raw_outputs = [g["raw_output"] for g in gen_outputs]
    decisions = extractor.extract_batch(raw_outputs)
    
    # 2. Claim extraction + NLI verification
    verif_results = []
    for i, g in enumerate(gen_outputs):
        claims = claim_extractor.extract(g["raw_output"])
        verif = nli_verifier.verify_answer(
            claims=claims,
            evidence_passages=g.get("evidence_texts", []),
            question_pmid=g.get("question_pmid", ""),
        )
        verif_results.append(verif.to_dict())
        if (i + 1) % 10 == 0 or (i + 1) == len(gen_outputs):
            print(f"  Verified {i+1}/{len(gen_outputs)} answers")
            
    verif_path = resolve_path(cfg["results"]["base_dir"]) / "verification" / f"{m}_val_nli.jsonl"
    save_jsonl(verif_results, verif_path)
    processed_data[m] = {
        "gen_outputs": gen_outputs,
        "decisions": decisions,
        "verif_results": verif_results,
    }

nli_verifier.unload_model()
del nli_verifier
extractor.unload_classifier()
del extractor
clear_gpu_memory()
print("✓ Claim verification complete!")
""")

    # ── CELL 12: Step 7 Trust Feature Construction ──────────────────────────────
    add_md("## Step 7: Trust Feature Construction (9 Features)\nConstructs the 9-dimensional trust feature vectors combining retrieval confidence, NLI entailment/contradiction, lexical overlap, and answer properties.")
    add_code("""import numpy as np
import pandas as pd
from src.trust_features import TrustFeatureBuilder, FEATURE_NAMES

feature_builder = TrustFeatureBuilder()
trust_matrices = {}

for m in model_keys:
    gen_outputs = processed_data[m]["gen_outputs"]
    decisions = processed_data[m]["decisions"]
    verif_results = processed_data[m]["verif_results"]
    
    pmid_to_gold = {str(r["pmid"]): r["final_decision"] for r in val_records}
    gold_labels = [pmid_to_gold.get(g["question_pmid"], "") for g in gen_outputs]
    pred_decisions = [d["decision"] for d in decisions]
    correct = np.array([1 if p.lower() == g.lower() else 0 for p, g in zip(pred_decisions, gold_labels)])
    
    tf_list = []
    for g, v in zip(gen_outputs, verif_results):
        ret_scores = g.get("retrieval_scores", [0.0])
        ret_dict = {
            "top1_score": ret_scores[0] if ret_scores else 0.0,
            "score_margin": (ret_scores[0] - ret_scores[1]) if len(ret_scores) > 1 else 0.0,
            "retrieved_pmids": g.get("evidence_pmids", []),
        }
        nli_dict = {
            "mean_max_entailment": v.get("mean_max_entailment", 0.0),
            "frac_entailed": v.get("frac_entailed", 0.0),
            "max_contradiction": v.get("max_contradiction", 0.0),
            "frac_contradicted": v.get("frac_contradicted", 0.0),
        }
        evidence_text = " ".join(g.get("evidence_texts", []))
        tf = feature_builder.build(
            retrieval_result=ret_dict,
            nli_result=nli_dict,
            generated_answer=g["raw_output"],
            evidence_text=evidence_text,
            question=g["question"],
            question_pmid=g["question_pmid"],
        )
        tf_list.append(tf)
        
    X = feature_builder.get_feature_matrix(tf_list)
    trust_matrices[m] = {
        "X": X,
        "y": correct,
        "tf_list": tf_list,
        "pred_decisions": pred_decisions,
        "gold_labels": gold_labels,
    }
    print(f"✓ {m.upper()}: Feature matrix {X.shape}, Empirical Accuracy: {correct.mean():.3f}")

# Display feature sample
df_feat_sample = pd.DataFrame(trust_matrices["qwen"]["X"][:5], columns=FEATURE_NAMES)
print("\\nSample 9-Feature Vectors (Qwen):")
display(df_feat_sample)
""")

    # ── CELL 13: Step 8 Calibrated Trust Model ──────────────────────────────────
    add_md("## Step 8: Calibrated Trust Model Training & Feature Weights\nTrains L2-regularized Logistic Regression with Platt scaling calibration and extracts feature log-odds weights.")
    add_code("""from src.trust_model import TrustModel

trained_models = {}
for m in model_keys:
    X = trust_matrices[m]["X"]
    y = trust_matrices[m]["y"]
    
    model = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        cv_folds=min(5, max(2, len(y) // 2)),
        calibration_method="platt",
        seed=42,
    )
    
    if len(np.unique(y)) > 1:
        split_idx = max(2, int(len(X) * 0.7))
        X_tr, y_tr = X[:split_idx], y[:split_idx]
        X_cal, y_cal = X[split_idx:], y[split_idx:]
        train_metrics = model.fit(X_tr, y_tr, X_val=X_cal, y_val=y_cal)
        print(f"\\n--- {m.upper()} Calibrated Trust Model ---")
        print(f"CV AUROC: {train_metrics.get('cv_auroc_mean', 'N/A')}")
        if "coefficients" in train_metrics:
            df_coef = pd.DataFrame(
                list(train_metrics["coefficients"].items()),
                columns=["Feature", "Log-Odds Weight"]
            ).sort_values(by="Log-Odds Weight", key=abs, ascending=False)
            display(df_coef)
        trust_scores = model.predict_trust(X)
    else:
        # Fallback to heuristic if only single class exists in demo slice
        print(f"Notice: Single class in sample slice — using Heuristic Trust for {m.upper()}")
        heuristic = TrustModel(model_type="heuristic")
        heuristic._is_fitted = True
        trust_scores = heuristic.predict_trust(X)
        model = heuristic
        
    trained_models[m] = model
    trust_matrices[m]["trust_scores"] = trust_scores
""")

    # ── CELL 14: Step 9 Full Pipeline Evaluation ────────────────────────────────
    add_md("## Step 9: Pipeline Evaluation & Decorrelation Analysis\nComputes Correctness (Accuracy, Macro-F1), Trust Calibration (ECE, Brier), Discrimination (AUROC), and the 2×2 Trust-Correctness Contingency Matrix.")
    add_code("""from src.evaluation import Evaluator
from src.utils import save_json

evaluator = Evaluator.from_config(cfg)
eval_results = {}

for m in model_keys:
    pred_decisions = trust_matrices[m]["pred_decisions"]
    gold_labels = trust_matrices[m]["gold_labels"]
    trust_scores = trust_matrices[m]["trust_scores"]
    
    res = evaluator.evaluate_all(pred_decisions, gold_labels, trust_scores)
    eval_results[m] = res
    save_json(res, resolve_path(cfg["results"]["base_dir"]) / "final" / f"{m}_val_evaluation.json")
    
    print(f"\\n{'='*55}")
    print(f"{m.upper()} EVALUATION REPORT")
    print(f"{'='*55}")
    c = res["correctness"]
    print(f"Accuracy: {c['accuracy']:.4f} (95% CI: [{c['accuracy_ci'][0]:.4f}, {c['accuracy_ci'][1]:.4f}])")
    print(f"Macro F1: {c['macro_f1']:.4f}")
    
    cal = res.get("calibration", {})
    print(f"ECE (Expected Calibration Error): {cal.get('ece', 'N/A')}")
    print(f"Brier Score: {cal.get('brier_score', 'N/A')}")
    
    disc = res.get("discrimination", {})
    print(f"AUROC: {disc.get('auroc', 'N/A')}")
    
    dec = res.get("decorrelation", {})
    rates = dec.get("rates", {})
    print(f"Dangerous Rate (High Trust + Incorrect): {rates.get('dangerous_rate', 'N/A')}")
    print(f"Lucky Rate (Low Trust + Correct):        {rates.get('lucky_rate', 'N/A')}")
""")

    # ── CELL 15: Step 10 Error Taxonomy ─────────────────────────────────────────
    add_md("## Step 10: 6-Category Error Taxonomy Analysis\nClassifies all incorrect answers into E1 (Retrieval Failure) through E6 (Misinterpretation).")
    add_code("""from src.error_analysis import ErrorAnalyzer
from src.utils import save_json

analyzer = ErrorAnalyzer()
error_results = {}
ret_cache = load_jsonl(resolve_path(cfg["results"]["base_dir"]) / "retrieval" / "val_retrieval.jsonl")
pmid_to_ret = {r["question_pmid"]: r for r in ret_cache}

for m in model_keys:
    gen_outputs = processed_data[m]["gen_outputs"]
    decisions = trust_matrices[m]["pred_decisions"]
    gold_labels = trust_matrices[m]["gold_labels"]
    correct = trust_matrices[m]["y"]
    verif = processed_data[m]["verif_results"]
    
    error_inputs = []
    for i, g in enumerate(gen_outputs):
        pmid = g["question_pmid"]
        ret_item = pmid_to_ret.get(pmid, {})
        ret_pmids = ret_item.get("evidence_pmids", [])
        
        error_inputs.append({
            "question_pmid": pmid,
            "question": g["question"],
            "correct": bool(correct[i]),
            "predicted_decision": decisions[i],
            "gold_decision": gold_labels[i],
            "gold_pmid_in_top5": pmid in ret_pmids,
            "gold_pmid_rank": (ret_pmids.index(pmid) + 1) if pmid in ret_pmids else 0,
            "mean_max_entailment": verif[i].get("mean_max_entailment", 0.0),
            "max_contradiction": verif[i].get("max_contradiction", 0.0),
        })
        
    err_res = analyzer.classify_errors(error_inputs)
    error_results[m] = err_res
    save_json(err_res, resolve_path(cfg["results"]["base_dir"]) / "final" / f"{m}_val_errors.json")
    
    print(f"\\n{m.upper()} Error Breakdown (Total Errors = {err_res['n_incorrect']}):")
    for cat, cnt in err_res["distribution"].items():
        pct = (cnt / err_res["n_incorrect"] * 100) if err_res["n_incorrect"] > 0 else 0
        print(f"  {cat:25s}: {cnt:2d} ({pct:5.1f}%)")
""")

    # ── CELL 16: Step 11 Ablation Studies ───────────────────────────────────────
    add_md("## Step 11: Ablation Studies\nRuns systematic ablations across retrieval modes, trust estimators, and feature subsets.")
    add_code("""from experiments.run_ablation import run_all_ablations
import pandas as pd

ablation_summary = run_all_ablations(
    config=cfg,
    model_key="qwen",
    split="val",
    skip_retrieval=False,
)

if "retrieval_ablation" in ablation_summary:
    print("\\n--- Retrieval Mode Ablation ---")
    display(pd.DataFrame(ablation_summary["retrieval_ablation"]).T[["recall_at_1", "recall_at_3", "recall_at_5", "mrr"]])

if "trust_model_ablation" in ablation_summary:
    print("\\n--- Trust Estimator Architecture Ablation ---")
    display(pd.DataFrame(ablation_summary["trust_model_ablation"]).T)

if "feature_ablation" in ablation_summary:
    print("\\n--- Feature Group Ablation ---")
    display(pd.DataFrame(ablation_summary["feature_ablation"]).T)
""")

    # ── CELL 17: Step 12 Figures & Tables ───────────────────────────────────────
    add_md("## Step 12: Publication-Ready Comparison Figures & Tables\nGenerates all 300-DPI publication figures and exports Markdown, LaTeX, and CSV benchmark tables.")
    add_code("""from experiments.generate_figures import run_all_figures
from IPython.display import Image, display

run_all_figures(config=cfg, split="val")

fig_dir = resolve_path(cfg["results"]["base_dir"]) / "figures"
final_dir = resolve_path(cfg["results"]["base_dir"]) / "final"

# Display Benchmark Comparison Table
md_table_path = final_dir / "model_comparison_table.md"
if md_table_path.exists():
    with open(md_table_path) as f:
        print(f.read())

print("\\nPublication Figures:")
for fig_name in [
    "figure1_reliability_qwen.png",
    "figure2_roc_pr_qwen.png",
    "figure3_contingency_qwen.png",
    "figure4_errors_qwen.png",
    "figure5_feature_importance.png",
    "figure6_confusion_qwen.png",
]:
    p = fig_dir / fig_name
    if p.exists():
        print(f"\\n--- {fig_name} ---")
        display(Image(filename=str(p), width=550))
""")

    # ── CELL 18: Step 13 Export Results ─────────────────────────────────────────
    add_md("## Step 13: Export and Download Results\nArchives all experiment outputs, checkpoints, trained models, figures, and tables into a single ZIP file.")
    add_code("""import shutil

zip_filename = "trust_rag_colab_results.zip"
results_dir = resolve_path(cfg["results"]["base_dir"])

shutil.make_archive("trust_rag_colab_results", "zip", results_dir)
print(f"✓ Archive created: {zip_filename} ({os.path.getsize(zip_filename) / 1e6:.2f} MB)")

try:
    from google.colab import files
    files.download(zip_filename)
    print("✓ Triggered browser download!")
except Exception as e:
    print(f"Download manually from Colab files panel: {zip_filename}")
""")

    return nb

if __name__ == "__main__":
    notebook = create_notebook()
    output_path = Path("notebooks/TRUST_RAG_Colab_Pipeline.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully generated {output_path} with {len(notebook['cells'])} cells.")
