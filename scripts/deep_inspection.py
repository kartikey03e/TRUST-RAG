#!/usr/bin/env python3
"""
Deep inspection script for TRUST-RAG using repository's TrustModel and Evaluator.
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, binomtest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.trust_features import FEATURE_NAMES
from src.trust_model import TrustModel
from src.evaluation import Evaluator

def run_deep_inspection():
    evaluator = Evaluator()
    train_path = repo_root / "results" / "trust" / "qwen_train_trust.jsonl"
    val_path = repo_root / "results" / "trust" / "qwen_val_trust.jsonl"
    med_val_path = repo_root / "results" / "trust" / "medgemma_val_trust.jsonl"

    def load_data(p):
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    train_recs = load_data(train_path)
    val_recs = load_data(val_path)
    med_val_recs = load_data(med_val_path)

    print(f"Official FEATURE_NAMES order ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    def extract_X_y(recs, feats):
        X = np.array([[float(r[f]) for f in feats] for r in recs], dtype=np.float64)
        y = np.array([1 if r["correct"] else 0 for r in recs], dtype=np.int64)
        return X, y

    X_train_full, y_train_full = extract_X_y(train_recs, FEATURE_NAMES)
    X_val, y_val = extract_X_y(val_recs, FEATURE_NAMES)

    X_tr_sub, X_cal_sub, y_tr_sub, y_cal_sub = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    print("=" * 80)
    print("1. DEEP INVESTIGATION: SEMANTIC-ONLY MODEL RAW VS CALIBRATED")
    print("=" * 80)
    sem_names = ["answer_evidence_sim", "question_answer_sim"]
    sem_indices = [FEATURE_NAMES.index(f) for f in sem_names]
    sub_X_tr_sem = X_tr_sub[:, sem_indices]
    sub_X_cal_sem = X_cal_sub[:, sem_indices]
    sub_X_val_sem = X_val[:, sem_indices]

    # A. Uncalibrated Model on Semantic Features
    uncal_sem = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        seed=42
    )
    uncal_sem._feature_names = sem_names
    uncal_sem.fit(sub_X_tr_sem, y_tr_sub) # No calibration set
    scores_uncal_val = uncal_sem.predict_trust(sub_X_val_sem)
    disc_uncal = evaluator.compute_discrimination(scores_uncal_val, y_val)
    print(f"Uncalibrated Semantic Model on Validation:")
    print(f"  AUROC = {disc_uncal['auroc']:.4f}")
    print(f"  AUPRC = {disc_uncal['auprc']:.4f}")

    # Decision function on calibration set
    scaled_cal_sem = uncal_sem._scaler.transform(sub_X_cal_sem)
    df_cal = uncal_sem._model.decision_function(scaled_cal_sem)
    r_cal, p_val_cal = pointbiserialr(df_cal, y_cal_sub)
    print(f"Calibration set (N=150) point-biserial correlation between uncalibrated logit & label:")
    print(f"  r = {r_cal:.6f}, p = {p_val_cal:.6f}")

    # B. Platt-Calibrated Model on Semantic Features
    platt_sem = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="platt",
        seed=42
    )
    platt_sem._feature_names = sem_names
    platt_sem.fit(sub_X_tr_sem, y_tr_sub, X_val=sub_X_cal_sem, y_val=y_cal_sub)
    scores_platt_val = platt_sem.predict_trust(sub_X_val_sem)
    disc_platt = evaluator.compute_discrimination(scores_platt_val, y_val)
    cal_platt = evaluator.compute_calibration(scores_platt_val, y_val)
    print(f"Platt-Calibrated Semantic Model on Validation:")
    print(f"  AUROC = {disc_platt['auroc']:.4f}")
    print(f"  AUPRC = {disc_platt['auprc']:.4f}")
    print(f"  ECE = {cal_platt['ece']:.4f}")
    print(f"  Brier = {cal_platt['brier_score']:.4f}")

    # Inspect the underlying calibrator parameters in platt_sem._model
    calibrated_clf = platt_sem._model
    if hasattr(calibrated_clf, "calibrated_classifiers_"):
        for i, cc in enumerate(calibrated_clf.calibrated_classifiers_):
            for c_idx, c in enumerate(cc.calibrators):
                a_val = getattr(c, "a_", getattr(c, "a", None))
                b_val = getattr(c, "b_", getattr(c, "b", None))
                print(f"  Fold {i} sub-calibrator {c_idx}: slope a = {a_val}, intercept b = {b_val}")

    print("\n" + "=" * 80)
    print("2. DEEP INVESTIGATION: FULL 9-FEATURE MODEL VS W/O RETRIEVAL")
    print("=" * 80)
    # Full 9 features
    model_9 = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="platt",
        seed=42
    )
    model_9.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_9 = model_9.predict_trust(X_val)
    disc_9 = evaluator.compute_discrimination(scores_9, y_val)
    cal_9 = evaluator.compute_calibration(scores_9, y_val)
    print(f"All 9 Features: AUROC = {disc_9['auroc']:.4f}, AUPRC = {disc_9['auprc']:.4f}, ECE = {cal_9['ece']:.4f}")

    # w/o retrieval features (semantic + NLI)
    retrieval_names = {"retrieval_top1_score", "retrieval_score_margin", "n_unique_pmids"}
    no_ret_indices = [i for i, name in enumerate(FEATURE_NAMES) if name not in retrieval_names]
    model_noret = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="platt",
        seed=42
    )
    model_noret._feature_names = [FEATURE_NAMES[i] for i in no_ret_indices]
    model_noret.fit(X_tr_sub[:, no_ret_indices], y_tr_sub, X_val=X_cal_sub[:, no_ret_indices], y_val=y_cal_sub)
    scores_noret = model_noret.predict_trust(X_val[:, no_ret_indices])
    disc_noret = evaluator.compute_discrimination(scores_noret, y_val)
    cal_noret = evaluator.compute_calibration(scores_noret, y_val)
    print(f"Without Retrieval Features (n=6): AUROC = {disc_noret['auroc']:.4f}, AUPRC = {disc_noret['auprc']:.4f}, ECE = {cal_noret['ece']:.4f}")

    # w/o NLI features (retrieval + semantic)
    nli_names = {"mean_max_entailment", "frac_entailed_claims", "max_contradiction", "frac_contradicted_claims"}
    no_nli_indices = [i for i, name in enumerate(FEATURE_NAMES) if name not in nli_names]
    model_nonli = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="platt",
        seed=42
    )
    model_nonli._feature_names = [FEATURE_NAMES[i] for i in no_nli_indices]
    model_nonli.fit(X_tr_sub[:, no_nli_indices], y_tr_sub, X_val=X_cal_sub[:, no_nli_indices], y_val=y_cal_sub)
    scores_nonli = model_nonli.predict_trust(X_val[:, no_nli_indices])
    disc_nonli = evaluator.compute_discrimination(scores_nonli, y_val)
    cal_nonli = evaluator.compute_calibration(scores_nonli, y_val)
    print(f"Without NLI Features (n=5): AUROC = {disc_nonli['auroc']:.4f}, AUPRC = {disc_nonli['auprc']:.4f}, ECE = {cal_nonli['ece']:.4f}")

    print("\nFeature Coefficients in Full 9-Feature Model (Properly Aligned):")
    # Base model coefficients from model_9
    # Note: in TrustModel, if calibrated, base model is inside calibrated classifier or fitted on X_train_scaled
    # Let's inspect base model directly
    base_clf = LogisticRegression(C=1.0, random_state=42)
    scaler_9 = StandardScaler()
    X_tr_scaled = scaler_9.fit_transform(X_tr_sub)
    base_clf.fit(X_tr_scaled, y_tr_sub)
    for fn, coef in zip(FEATURE_NAMES, base_clf.coef_[0]):
        print(f"  {fn:28s}: {coef:+.4f}")
    print(f"  {'Intercept':28s}: {base_clf.intercept_[0]:+.4f}")

    print("\n" + "=" * 80)
    print("3. DEEP INVESTIGATION: HIGH-TRUST TIER COUNTS AT THRESHOLD tau = 0.5")
    print("=" * 80)
    # Check Platt-calibrated high trust
    high_trust_platt = scores_9 >= 0.5
    n_high_p = int(np.sum(high_trust_platt))
    corr_high_p = int(np.sum(y_val[high_trust_platt]))
    incorr_high_p = n_high_p - corr_high_p
    print(f"Platt-calibrated Model (tau >= 0.5):")
    print(f"  High-Trust Count = {n_high_p} / {len(y_val)} (Coverage: {n_high_p/len(y_val)*100:.2f}%)")
    print(f"  High-Trust Correct = {corr_high_p}, Incorrect = {incorr_high_p}")
    print(f"  High-Trust Accuracy = {corr_high_p / n_high_p * 100:.2f}% (based on n={n_high_p} sample!)")
    print(f"  Low-Trust Count = {len(y_val) - n_high_p}, Correct = {np.sum(y_val[~high_trust_platt])}, Incorrect = {np.sum((1-y_val)[~high_trust_platt])}")

    # Check Isotonic-calibrated high trust
    model_iso = TrustModel(
        model_type="logistic_regression",
        regularization_C=1.0,
        calibration_method="isotonic",
        seed=42
    )
    model_iso.fit(X_tr_sub, y_tr_sub, X_val=X_cal_sub, y_val=y_cal_sub)
    scores_iso = model_iso.predict_trust(X_val)
    high_trust_iso = scores_iso >= 0.5
    n_high_iso = int(np.sum(high_trust_iso))
    corr_high_iso = int(np.sum(y_val[high_trust_iso]))
    incorr_high_iso = n_high_iso - corr_high_iso
    print(f"\nIsotonic-calibrated Model (tau >= 0.5):")
    print(f"  High-Trust Count = {n_high_iso} / {len(y_val)} (Coverage: {n_high_iso/len(y_val)*100:.2f}%)")
    print(f"  High-Trust Correct = {corr_high_iso}, Incorrect = {incorr_high_iso}")
    print(f"  High-Trust Accuracy = {corr_high_iso / n_high_iso * 100:.2f}%")
    print(f"  In-Group Error Rate = {incorr_high_iso / n_high_iso * 100:.2f}%")
    print(f"  Cohort Dangerous Rate = {incorr_high_iso / len(y_val) * 100:.2f}%")

    print("\n" + "=" * 80)
    print("4. DEEP INVESTIGATION: PAIRED STATISTICAL COMPARISON ON 200 VALIDATION QUESTIONS")
    print("=" * 80)
    qwen_dict = {str(r["question_pmid"]): r for r in val_recs}
    med_dict = {str(r["question_pmid"]): r for r in med_val_recs}

    shared_pmids = [str(r["question_pmid"]) for r in val_recs if str(r["question_pmid"]) in med_dict]
    print(f"Number of shared validation PMIDs: {len(shared_pmids)}")

    a, b, c, d = 0, 0, 0, 0
    for pmid in shared_pmids:
        q_corr = qwen_dict[pmid]["correct"]
        m_corr = med_dict[pmid]["correct"]
        if q_corr and m_corr:
            a += 1
        elif q_corr and not m_corr:
            b += 1
        elif not q_corr and m_corr:
            c += 1
        else:
            d += 1

    print(f"Contingency: Both={a}, Qwen-only={b}, MedGemma-only={c}, Neither={d}")
    print(f"Discordant pairs (b+c): {b+c}")
    chi2_edwards = (abs(b - c) - 1.0)**2 / (b + c)
    b_test = binomtest(b, b + c, 0.5, alternative="two-sided")
    print(f"McNemar with Edwards continuity correction: chi2 = {chi2_edwards:.4f}")
    print(f"Exact Binomial two-sided test: p-value = {b_test.pvalue:.6e}")

if __name__ == "__main__":
    run_deep_inspection()
