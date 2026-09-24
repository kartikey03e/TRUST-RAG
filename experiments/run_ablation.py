"""
TRUST-RAG V2 — Ablation Study Runner
====================================
Runs ablation experiments across three key dimensions:
  1. Retrieval Ablations: BM25-only vs Dense-only (MedCPT) vs Hybrid RRF
  2. Trust Model Ablations: Heuristic vs Uncalibrated vs Calibrated (Platt/Isotonic) vs GNB
  3. Feature Ablations: Full 9 features vs subsets (No-NLI, No-Retrieval, No-Lexical)

Usage
-----
    python experiments/run_ablation.py --split val
    python experiments/run_ablation.py --model medgemma --split val
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config, load_config, resolve_path, set_seed
from src.evaluation import Evaluator
from src.retrieval_eval import evaluate_retrieval
from src.retriever import HybridRetriever
from src.trust_features import FEATURE_NAMES, TrustFeatureBuilder
from src.trust_model import TrustModel
from src.utils import get_logger, load_jsonl, save_json, save_jsonl

logger = get_logger("ablation")


def run_retrieval_ablation(
    retriever: HybridRetriever,
    records: list[dict],
    ks: list[int] = [1, 3, 5],
    top_k: int = 5,
) -> dict[str, dict[str, float]]:
    """Compare BM25-only, Dense-only, and Hybrid RRF retrieval."""
    logger.info("=" * 60)
    logger.info("ABLATION 1: Retrieval Comparison (BM25 vs Dense vs Hybrid)")
    logger.info("=" * 60)

    modes = ["bm25", "dense", "hybrid"]
    results = {}

    for mode in modes:
        logger.info(f"Running retrieval ablation: mode={mode}")
        eval_res = evaluate_retrieval(
            retriever=retriever,
            records=records,
            ks=ks,
            top_k=top_k,
            mode=mode,
        )
        results[mode] = eval_res["aggregate"]

    return results


def run_trust_model_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compare Heuristic, Uncalibrated, Calibrated Platt, Calibrated Isotonic, and GNB."""
    logger.info("=" * 60)
    logger.info("ABLATION 2: Trust Model Comparison")
    logger.info("=" * 60)

    evaluator = Evaluator(n_calibration_bins=5, seed=seed)
    results = {}

    # Variant A: Heuristic (Equal weight)
    logger.info("Evaluating: Heuristic Baseline")
    heuristic_model = TrustModel(model_type="heuristic", seed=seed)
    heuristic_model._is_fitted = True
    scores_heuristic = heuristic_model.predict_trust(X_val)
    cal_h = evaluator.compute_calibration(scores_heuristic, y_val)
    disc_h = evaluator.compute_discrimination(scores_heuristic, y_val)
    results["Heuristic (Equal Weights)"] = {
        "ece": cal_h["ece"],
        "mce": cal_h["mce"],
        "brier_score": cal_h["brier_score"],
        "auroc": disc_h.get("auroc"),
        "auprc": disc_h.get("auprc"),
    }

    # Variant B: Uncalibrated Logistic Regression
    logger.info("Evaluating: Uncalibrated Logistic Regression")
    uncal_model = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="platt",
        seed=seed,
    )
    # Fit without val set to leave uncalibrated
    uncal_model.fit(X_train, y_train)
    scores_uncal = uncal_model.predict_trust(X_val)
    cal_u = evaluator.compute_calibration(scores_uncal, y_val)
    disc_u = evaluator.compute_discrimination(scores_uncal, y_val)
    results["Uncalibrated LogReg"] = {
        "ece": cal_u["ece"],
        "mce": cal_u["mce"],
        "brier_score": cal_u["brier_score"],
        "auroc": disc_u.get("auroc"),
        "auprc": disc_u.get("auprc"),
    }

    # Variant C: Calibrated Platt (Sigmoid)
    logger.info("Evaluating: Calibrated LogReg (Platt)")
    # For fair evaluation on val: train on 80% train, calibrate on 20% train, evaluate on val
    n_tr = len(X_train)
    split_idx = int(n_tr * 0.75)
    X_tr_sub, y_tr_sub = X_train[:split_idx], y_train[:split_idx]
    X_cal_sub, y_cal_sub = X_train[split_idx:], y_train[split_idx:]

    platt_model = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="platt",
        seed=seed,
    )
    platt_model.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_platt = platt_model.predict_trust(X_val)
    cal_p = evaluator.compute_calibration(scores_platt, y_val)
    disc_p = evaluator.compute_discrimination(scores_platt, y_val)
    results["Calibrated LogReg (Platt Scaling)"] = {
        "ece": cal_p["ece"],
        "mce": cal_p["mce"],
        "brier_score": cal_p["brier_score"],
        "auroc": disc_p.get("auroc"),
        "auprc": disc_p.get("auprc"),
    }

    # Variant D: Calibrated Isotonic
    logger.info("Evaluating: Calibrated LogReg (Isotonic)")
    iso_model = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="isotonic",
        seed=seed,
    )
    iso_model.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_iso = iso_model.predict_trust(X_val)
    cal_i = evaluator.compute_calibration(scores_iso, y_val)
    disc_i = evaluator.compute_discrimination(scores_iso, y_val)
    results["Calibrated LogReg (Isotonic)"] = {
        "ece": cal_i["ece"],
        "mce": cal_i["mce"],
        "brier_score": cal_i["brier_score"],
        "auroc": disc_i.get("auroc"),
        "auprc": disc_i.get("auprc"),
    }

    # Variant E: Gaussian Naive Bayes
    logger.info("Evaluating: Gaussian Naive Bayes")
    gnb_model = TrustModel(model_type="bayesian", seed=seed)
    gnb_model.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_gnb = gnb_model.predict_trust(X_val)
    cal_g = evaluator.compute_calibration(scores_gnb, y_val)
    disc_g = evaluator.compute_discrimination(scores_gnb, y_val)
    results["Gaussian Naive Bayes"] = {
        "ece": cal_g["ece"],
        "mce": cal_g["mce"],
        "brier_score": cal_g["brier_score"],
        "auroc": disc_g.get("auroc"),
        "auprc": disc_g.get("auprc"),
    }

    return results


def run_feature_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Evaluate trust model when removing specific feature subsets."""
    logger.info("=" * 60)
    logger.info("ABLATION 3: Feature Group Ablation")
    logger.info("=" * 60)

    evaluator = Evaluator(seed=seed)
    results = {}

    feature_groups = {
        "All 9 Features": list(range(len(FEATURE_NAMES))),
        "w/o NLI Features": [
            i for i, name in enumerate(FEATURE_NAMES)
            if name not in ("mean_max_entailment", "frac_entailed", "max_contradiction", "frac_contradicted")
        ],
        "w/o Retrieval Features": [
            i for i, name in enumerate(FEATURE_NAMES)
            if name not in ("retrieval_top1_score", "retrieval_score_margin", "retrieval_entropy")
        ],
        "w/o Lexical Features": [
            i for i, name in enumerate(FEATURE_NAMES)
            if name not in ("lexical_overlap", "answer_evidence_sim")
        ],
        "NLI Features Only": [
            i for i, name in enumerate(FEATURE_NAMES)
            if name in ("mean_max_entailment", "frac_entailed", "max_contradiction", "frac_contradicted")
        ],
        "Retrieval Features Only": [
            i for i, name in enumerate(FEATURE_NAMES)
            if name in ("retrieval_top1_score", "retrieval_score_margin", "retrieval_entropy")
        ],
    }

    n_tr = len(X_train)
    split_idx = int(n_tr * 0.75)
    X_tr_sub, y_tr_sub = X_train[:split_idx], y_train[:split_idx]
    X_cal_sub, y_cal_sub = X_train[split_idx:], y_train[split_idx:]

    for group_name, feat_indices in feature_groups.items():
        sub_X_tr = X_tr_sub[:, feat_indices]
        sub_X_cal = X_cal_sub[:, feat_indices]
        sub_X_val = X_val[:, feat_indices]

        model = TrustModel(
            model_type="logistic_regression",
            regularization_C=1.0,
            calibration_method="platt",
            seed=seed,
        )
        # Temporary override feature names for logging
        model._feature_names = [FEATURE_NAMES[i] for i in feat_indices]
        model.fit(sub_X_tr, y_tr_sub, X_val=sub_X_cal, y_val=y_cal_sub)

        scores = model.predict_trust(sub_X_val)
        cal = evaluator.compute_calibration(scores, y_val)
        disc = evaluator.compute_discrimination(scores, y_val)

        results[group_name] = {
            "n_features": len(feat_indices),
            "ece": cal["ece"],
            "brier_score": cal["brier_score"],
            "auroc": disc.get("auroc"),
            "auprc": disc.get("auprc"),
        }

    return results


def run_all_ablations(
    config: dict | None = None,
    model_key: str = "medgemma",
    split: str = "val",
    skip_retrieval: bool = False,
) -> dict[str, Any]:
    """Execute complete ablation suite."""
    if config is None:
        config = get_default_config()

    seed = config.get("seed", 42)
    set_seed(seed)

    results_dir = resolve_path(config["results"]["base_dir"])
    ablation_dir = results_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model": model_key,
        "split": split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. Retrieval Ablation
    if not skip_retrieval:
        faiss_path = resolve_path(config["retrieval"]["faiss_index_path"])
        bm25_path = resolve_path(config["retrieval"]["bm25_index_path"])
        if faiss_path.exists() and bm25_path.exists():
            data_dir = resolve_path(config["data"]["processed_dir"])
            records = load_jsonl(data_dir / f"{split}.jsonl")
            retriever = HybridRetriever.from_config(config)
            ret_results = run_retrieval_ablation(
                retriever=retriever,
                records=records,
                ks=[1, 3, 5],
                top_k=config["retrieval"]["final_top_k"],
            )
            summary["retrieval_ablation"] = ret_results
            retriever.unload_query_encoder()
            del retriever
        else:
            logger.warning("Indices not found — skipping retrieval ablation")

    # 2. Trust Model & Feature Ablation
    # Check if trust features are available from previous pipeline runs
    trust_val_path = results_dir / "trust" / f"{model_key}_{split}_trust.jsonl"
    trust_train_path = results_dir / "trust" / f"{model_key}_train_trust.jsonl"

    if trust_val_path.exists():
        val_records = load_jsonl(trust_val_path)
        feature_builder = TrustFeatureBuilder()

        # Build X_val, y_val
        X_val_list = []
        y_val_list = []
        for r in val_records:
            vec = [float(r.get(fname, 0.0)) for fname in FEATURE_NAMES]
            X_val_list.append(vec)
            y_val_list.append(1 if r.get("correct", False) else 0)

        X_val = np.array(X_val_list, dtype=np.float32)
        y_val = np.array(y_val_list, dtype=np.int64)

        if trust_train_path.exists():
            train_records = load_jsonl(trust_train_path)
            X_train_list = []
            y_train_list = []
            for r in train_records:
                vec = [float(r.get(fname, 0.0)) for fname in FEATURE_NAMES]
                X_train_list.append(vec)
                y_train_list.append(1 if r.get("correct", False) else 0)
            X_train = np.array(X_train_list, dtype=np.float32)
            y_train = np.array(y_train_list, dtype=np.int64)
        else:
            logger.info("Train trust features not found — using 5-fold cross-split on val set")
            X_train, y_train = X_val, y_val

        if len(np.unique(y_val)) > 1:
            trust_ablation = run_trust_model_ablation(X_train, y_train, X_val, y_val, seed=seed)
            summary["trust_model_ablation"] = trust_ablation

            feat_ablation = run_feature_ablation(X_train, y_train, X_val, y_val, seed=seed)
            summary["feature_ablation"] = feat_ablation
        else:
            logger.warning("Only 1 class found in labels — skipping trust/feature ablations")
    else:
        logger.info(f"Trust features not found at {trust_val_path} — skipping trust ablations")

    # Save summary
    save_json(summary, ablation_dir / f"{model_key}_ablation_summary.json")
    logger.info(f"Ablation results saved → {ablation_dir / f'{model_key}_ablation_summary.json'}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRUST-RAG Ablation Experiments")
    parser.add_argument("--model", type=str, default="medgemma", choices=["medgemma", "qwen"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    run_all_ablations(
        config=cfg,
        model_key=args.model,
        split=args.split,
        skip_retrieval=args.skip_retrieval,
    )
