"""
TRUST-RAG: Train Trust Estimators on Train Split & Evaluate on Untouched Val Split
==================================================================================
1. Loads 150-sample training trust features from results/trust/qwen_train_trust.jsonl
2. Splits training data into 75% train-sub and 25% calibration-sub (stratified).
3. Fits Heuristic, Uncalibrated LogReg, Platt LogReg, Isotonic LogReg, and GNB.
4. Evaluates all models strictly on the untouched 200 validation samples (results/trust/qwen_val_trust.jsonl).
5. Computes Discrimination (AUROC, AUPRC), Calibration (ECE, MCE, Brier), and Trust-Correctness Contingency tables.
6. Generates high-resolution 300 DPI publication figures.
7. Exports results to results/final/trust_model_comparison_verified.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import Evaluator
from src.trust_features import FEATURE_NAMES
from src.trust_model import TrustModel
from src.utils import get_logger, load_jsonl, save_json

logger = get_logger("trust_eval_pipeline")

def run_pipeline():
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_dir = results_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Training Data
    train_path = results_dir / "trust" / "qwen_train_trust.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    train_records = load_jsonl(train_path)
    logger.info(f"Loaded {len(train_records)} training records from {train_path}")

    # 2. Load Validation Data
    val_path = results_dir / "trust" / "qwen_val_trust.jsonl"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}")
    val_records = load_jsonl(val_path)
    logger.info(f"Loaded {len(val_records)} validation records from {val_path}")

    # Convert to matrices
    X_train = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in train_records], dtype=np.float32)
    y_train = np.array([1 if r["correct"] else 0 for r in train_records], dtype=np.int64)

    X_val = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in val_records], dtype=np.float32)
    y_val = np.array([1 if r["correct"] else 0 for r in val_records], dtype=np.int64)

    seed = 42
    evaluator = Evaluator(n_calibration_bins=5, seed=seed)

    # Stratified split of training set for calibration (450 fit, 150 calibrate)
    train_indices = np.arange(len(train_records))
    idx_tr_sub, idx_cal_sub = train_test_split(
        train_indices, test_size=0.25, random_state=seed, stratify=y_train
    )
    X_tr_sub, y_tr_sub = X_train[idx_tr_sub], y_train[idx_tr_sub]
    X_cal_sub, y_cal_sub = X_train[idx_cal_sub], y_train[idx_cal_sub]
    logger.info(f"Train partition: {len(X_tr_sub)} fit, {len(X_cal_sub)} calibrate. Validation: {len(X_val)} untouched.")

    # Save split metadata for complete reproducibility
    split_meta = {
        "dataset": "PubMedQA (pqal)",
        "random_seed": seed,
        "n_train_total": len(train_records),
        "n_fit_subsplit": len(idx_tr_sub),
        "n_calibration_subsplit": len(idx_cal_sub),
        "n_validation_untouched": len(val_records),
        "fit_subsplit_pmids": [train_records[i]["question_pmid"] for i in idx_tr_sub],
        "calibration_subsplit_pmids": [train_records[i]["question_pmid"] for i in idx_cal_sub],
        "validation_untouched_pmids": [r["question_pmid"] for r in val_records],
    }
    save_json(split_meta, final_dir / "train_calibration_split_metadata.json")
    logger.info(f"Saved split metadata to {final_dir / 'train_calibration_split_metadata.json'}")

    # -------------------------------------------------------------
    # Model 1: Heuristic Baseline (Equal Weights)
    # -------------------------------------------------------------
    heu_model = TrustModel(model_type="heuristic", seed=seed)
    heu_model._is_fitted = True
    scores_heu = heu_model.predict_trust(X_val)

    # -------------------------------------------------------------
    # Model 2: Uncalibrated Logistic Regression (Fit on 450-sample fitting partition)
    # -------------------------------------------------------------
    uncal_model = TrustModel(model_type="logistic_regression", regularization_C=1.0, seed=seed)
    train_metrics = uncal_model.fit(X_tr_sub, y_tr_sub)
    scores_uncal = uncal_model.predict_trust(X_val)
    feature_coefficients = train_metrics.get("coefficients", {})
    intercept = train_metrics.get("intercept", 0.0)

    # -------------------------------------------------------------
    # Model 3: Platt-Calibrated Logistic Regression (Fit tr_sub, cal on cal_sub)
    # -------------------------------------------------------------
    platt_model = TrustModel(model_type="logistic_regression", regularization_C=1.0, calibration_method="platt", seed=seed)
    platt_model.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_platt = platt_model.predict_trust(X_val)

    # -------------------------------------------------------------
    # Model 4: Isotonic-Calibrated Logistic Regression (Fit tr_sub, cal on cal_sub)
    # -------------------------------------------------------------
    iso_model = TrustModel(model_type="logistic_regression", regularization_C=1.0, calibration_method="isotonic", seed=seed)
    iso_model.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_iso = iso_model.predict_trust(X_val)

    # -------------------------------------------------------------
    # Model 5: Gaussian Naive Bayes (Fit tr_sub, cal on cal_sub)
    # -------------------------------------------------------------
    gnb_model = TrustModel(model_type="bayesian", seed=seed)
    gnb_model.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_gnb = gnb_model.predict_trust(X_val)

    models_dict = {
        "Heuristic (Equal Weights)": scores_heu,
        "Uncalibrated LogReg": scores_uncal,
        "Calibrated LogReg (Platt)": scores_platt,
        "Calibrated LogReg (Isotonic)": scores_iso,
        "Gaussian Naive Bayes": scores_gnb,
    }

    results = {}
    for name, scores in models_dict.items():
        cal = evaluator.compute_calibration(scores, y_val)
        disc = evaluator.compute_discrimination(scores, y_val)
        decorr = evaluator.compute_decorrelation(scores, y_val, trust_threshold=0.5)

        results[name] = {
            "discrimination": {
                "auroc": disc.get("auroc"),
                "auprc": disc.get("auprc"),
            },
            "calibration": {
                "ece": cal["ece"],
                "mce": cal["mce"],
                "brier_score": cal["brier_score"],
                "reliability_diagram": cal["reliability_diagram"],
            },
            "contingency_table": decorr["contingency_table"],
            "rates": decorr["rates"],
        }

    full_output = {
        "dataset": "PubMedQA (pqal)",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "models": results,
        "learned_coefficients": feature_coefficients,
        "intercept": intercept,
    }

    save_json(full_output, final_dir / "trust_model_comparison_verified.json")
    logger.info(f"Saved verified comparison to {final_dir / 'trust_model_comparison_verified.json'}")

    # -------------------------------------------------------------
    # PLOT FIGURES AT 300 DPI
    # -------------------------------------------------------------
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

    # --- Figure 1: Reliability Diagrams ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    for ax, m_name, col in zip([ax1, ax2], ["Heuristic (Equal Weights)", "Calibrated LogReg (Isotonic)"], ["#e74c3c", "#2ecc71"]):
        rd = results[m_name]["calibration"]["reliability_diagram"]
        ece_val = results[m_name]["calibration"]["ece"]
        brier_val = results[m_name]["calibration"]["brier_score"]
        confs = rd["bin_confidences"]
        accs = rd["bin_accuracies"]
        counts = rd["bin_counts"]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
        ax.plot(confs, accs, marker="s", color=col, linewidth=2, markersize=7, label=f"{m_name}\n(ECE={ece_val:.4f}, Brier={brier_val:.4f})")
        for x, y, c in zip(confs, accs, counts):
            if c > 0:
                ax.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title(m_name)
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig1_path = figures_dir / "fig1_reliability_comparison_verified.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig1_path}")

    # --- Figure 2: Discrimination Curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    palette = {
        "Heuristic (Equal Weights)": "#95a5a6",
        "Uncalibrated LogReg": "#3498db",
        "Calibrated LogReg (Platt)": "#e67e22",
        "Calibrated LogReg (Isotonic)": "#2ecc71",
        "Gaussian Naive Bayes": "#9b59b6",
    }
    for m_name, sc in models_dict.items():
        fpr, tpr, _ = roc_curve(y_val, sc)
        roc_auc = roc_auc_score(y_val, sc)
        prec, rec, _ = precision_recall_curve(y_val, sc)
        pr_auc = auc(rec, prec)

        ax1.plot(fpr, tpr, color=palette[m_name], label=f"{m_name} ({roc_auc:.3f})", linewidth=1.8)
        ax2.plot(rec, prec, color=palette[m_name], label=f"{m_name} ({pr_auc:.3f})", linewidth=1.8)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (0.500)")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("ROC Curves on Untouched Val Split")
    ax1.legend(loc="lower right", fontsize=8)

    pos_ratio = y_val.mean()
    ax2.axhline(pos_ratio, color="k", linestyle="--", alpha=0.5, label=f"No Skill ({pos_ratio:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves on Untouched Val Split")
    ax2.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig2_path = figures_dir / "fig2_discrimination_curves_verified.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig2_path}")

    # --- Figure 3: Feature Coefficients ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    feat_names = list(feature_coefficients.keys())
    weights = [feature_coefficients[k] for k in feat_names]
    sorted_idx = np.argsort(weights)
    sorted_names = [feat_names[i] for i in sorted_idx]
    sorted_weights = [weights[i] for i in sorted_idx]
    bar_colors = ["#e74c3c" if w < 0 else "#2ecc71" for w in sorted_weights]

    ax.barh(sorted_names, sorted_weights, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Standardized Logistic Regression Weight")
    ax.set_title("Learned Feature Importance (Trained on Training Split)")
    for i, w in enumerate(sorted_weights):
        offset = 0.008 if w >= 0 else -0.008
        ha = "left" if w >= 0 else "right"
        ax.annotate(f"{w:+.4f}", (w + offset, i), va="center", ha=ha, fontsize=9)
    ax.set_xlim(min(sorted_weights) - 0.05, max(sorted_weights) + 0.06)

    plt.tight_layout()
    fig3_path = figures_dir / "fig3_feature_coefficients_verified.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig3_path}")

    # Also update figure5_feature_importance.png
    fig5_orig = figures_dir / "figure5_feature_importance.png"
    plt.figure(figsize=(8, 5), dpi=300)
    plt.barh(sorted_names, sorted_weights, color=bar_colors, edgecolor="black", linewidth=0.6)
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
    plt.xlabel("Standardized Logistic Regression Weight")
    plt.title("Learned Feature Importance (Trained on Training Split)")
    for i, w in enumerate(sorted_weights):
        offset = 0.008 if w >= 0 else -0.008
        ha = "left" if w >= 0 else "right"
        plt.annotate(f"{w:+.4f}", (w + offset, i), va="center", ha=ha, fontsize=9)
    plt.xlim(min(sorted_weights) - 0.05, max(sorted_weights) + 0.06)
    plt.tight_layout()
    plt.savefig(fig5_orig, dpi=300)
    plt.close()

    print("\n" + "="*70)
    print("VERIFIED TRUST ESTIMATION PIPELINE RESULTS")
    print("="*70)
    print(f"{'Model':<30} | {'AUROC':<7} | {'AUPRC':<7} | {'ECE':<7} | {'Brier':<7} | {'Dang %':<7}")
    print("-" * 75)
    for m_name, d in results.items():
        print(f"{m_name:<30} | {d['discrimination']['auroc']:<7.4f} | {d['discrimination']['auprc']:<7.4f} | {d['calibration']['ece']:<7.4f} | {d['calibration']['brier_score']:<7.4f} | {d['rates']['dangerous_rate']*100:<6.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
