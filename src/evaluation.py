"""
TRUST-RAG V2 — Evaluation
===========================
Computes correctness metrics, trust quality metrics, calibration analysis,
and trust-correctness decorrelation analysis.

Usage
-----
    from src.evaluation import Evaluator
    evaluator = Evaluator()
    results = evaluator.evaluate_all(predictions, trust_scores, gold_labels)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import get_default_config
from src.utils import get_logger, save_json

logger = get_logger(__name__)


class Evaluator:
    """Comprehensive evaluation for TRUST-RAG pipeline."""

    def __init__(
        self,
        n_calibration_bins: int = 5,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ):
        self.n_calibration_bins = n_calibration_bins
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed

    @classmethod
    def from_config(cls, config: dict | None = None) -> "Evaluator":
        if config is None:
            config = get_default_config()
        eval_cfg = config.get("evaluation", {})
        return cls(
            n_calibration_bins=eval_cfg.get("calibration_bins", 5),
            n_bootstrap=eval_cfg.get("bootstrap_samples", 1000),
            confidence_level=eval_cfg.get("confidence_level", 0.95),
            seed=config.get("seed", 42),
        )

    # ── Correctness metrics ──────────────────────────────────────────────

    def compute_correctness(
        self,
        predictions: list[str],
        gold_labels: list[str],
        labels: list[str] = ["yes", "no", "maybe"],
    ) -> dict[str, Any]:
        """Compute answer correctness metrics.

        Parameters
        ----------
        predictions : list[str]
            Predicted decisions.
        gold_labels : list[str]
            Gold PubMedQA labels.
        labels : list[str]
            Class labels.

        Returns
        -------
        dict
            Correctness metrics.
        """
        preds = [p.lower() for p in predictions]
        golds = [g.lower() for g in gold_labels]

        acc = accuracy_score(golds, preds)
        macro_f1 = f1_score(golds, preds, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(golds, preds, labels=labels, average="weighted", zero_division=0)

        # Per-class accuracy
        per_class_acc = {}
        for label in labels:
            mask = [g == label for g in golds]
            if sum(mask) > 0:
                label_preds = [p for p, m in zip(preds, mask) if m]
                label_golds = [g for g, m in zip(golds, mask) if m]
                per_class_acc[label] = round(accuracy_score(label_golds, label_preds), 4)
            else:
                per_class_acc[label] = None

        # Confusion matrix
        cm = confusion_matrix(golds, preds, labels=labels)

        # Prediction distribution
        from collections import Counter

        pred_dist = dict(Counter(preds))
        gold_dist = dict(Counter(golds))

        # Classification report
        report = classification_report(
            golds, preds, labels=labels, output_dict=True, zero_division=0
        )

        # Bootstrap CI for accuracy
        rng = np.random.RandomState(self.seed)
        boot_accs = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(preds), size=len(preds), replace=True)
            boot_preds = [preds[i] for i in idx]
            boot_golds = [golds[i] for i in idx]
            boot_accs.append(accuracy_score(boot_golds, boot_preds))
        alpha = (1 - self.confidence_level) / 2
        ci_low = float(np.percentile(boot_accs, alpha * 100))
        ci_high = float(np.percentile(boot_accs, (1 - alpha) * 100))

        result = {
            "accuracy": round(acc, 4),
            "accuracy_ci": [round(ci_low, 4), round(ci_high, 4)],
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm.tolist(),
            "confusion_labels": labels,
            "prediction_distribution": pred_dist,
            "gold_distribution": gold_dist,
            "classification_report": report,
            "n_samples": len(preds),
        }

        logger.info(f"Accuracy: {acc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        for label in labels:
            logger.info(f"  {label} acc: {per_class_acc.get(label, 'N/A')}")

        return result

    # ── Trust quality metrics ────────────────────────────────────────────

    def compute_calibration(
        self,
        trust_scores: np.ndarray,
        correct: np.ndarray,
    ) -> dict[str, Any]:
        """Compute calibration metrics (ECE, MCE, Brier, reliability diagram).

        Parameters
        ----------
        trust_scores : np.ndarray
            Trust scores in [0, 1], shape (n,).
        correct : np.ndarray
            Binary correctness labels, shape (n,).

        Returns
        -------
        dict
            Calibration metrics and bin data.
        """
        n = len(trust_scores)
        bins = np.linspace(0, 1, self.n_calibration_bins + 1)

        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(self.n_calibration_bins):
            mask = (trust_scores >= bins[i]) & (trust_scores < bins[i + 1])
            if i == self.n_calibration_bins - 1:
                mask = (trust_scores >= bins[i]) & (trust_scores <= bins[i + 1])

            count = mask.sum()
            if count > 0:
                bin_acc = correct[mask].mean()
                bin_conf = trust_scores[mask].mean()
            else:
                bin_acc = 0.0
                bin_conf = (bins[i] + bins[i + 1]) / 2

            bin_accs.append(float(bin_acc))
            bin_confs.append(float(bin_conf))
            bin_counts.append(int(count))

        # ECE: weighted average of |acc - conf| per bin
        ece = sum(
            (bin_counts[i] / n) * abs(bin_accs[i] - bin_confs[i])
            for i in range(self.n_calibration_bins)
            if bin_counts[i] > 0
        )

        # MCE: maximum calibration error
        mce = max(
            abs(bin_accs[i] - bin_confs[i])
            for i in range(self.n_calibration_bins)
            if bin_counts[i] > 0
        ) if any(c > 0 for c in bin_counts) else 0.0

        # Brier score
        brier = brier_score_loss(correct, trust_scores)

        result = {
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "brier_score": round(brier, 4),
            "reliability_diagram": {
                "bin_edges": bins.tolist(),
                "bin_accuracies": bin_accs,
                "bin_confidences": bin_confs,
                "bin_counts": bin_counts,
            },
        }

        logger.info(f"ECE: {ece:.4f} | MCE: {mce:.4f} | Brier: {brier:.4f}")
        return result

    def compute_discrimination(
        self,
        trust_scores: np.ndarray,
        correct: np.ndarray,
    ) -> dict[str, Any]:
        """Compute discrimination metrics (AUROC, AUPRC).

        Parameters
        ----------
        trust_scores : np.ndarray
            Trust scores.
        correct : np.ndarray
            Binary correctness labels.

        Returns
        -------
        dict
            AUROC, AUPRC, and curve data.
        """
        result: dict[str, Any] = {}

        # Check if both classes are present
        if len(np.unique(correct)) < 2:
            logger.warning("Only one class present — cannot compute AUROC/AUPRC")
            result["auroc"] = None
            result["auprc"] = None
            return result

        # AUROC
        auroc = roc_auc_score(correct, trust_scores)
        fpr, tpr, thresholds = roc_curve(correct, trust_scores)
        result["auroc"] = round(auroc, 4)
        result["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

        # AUPRC
        precision, recall, _ = precision_recall_curve(correct, trust_scores)
        auprc = auc(recall, precision)
        result["auprc"] = round(auprc, 4)

        logger.info(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
        return result

    # ── Trust-Correctness decorrelation ──────────────────────────────────

    def compute_decorrelation(
        self,
        trust_scores: np.ndarray,
        correct: np.ndarray,
        trust_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Analyze trust-correctness decorrelation.

        Produces the 2×2 contingency table and statistics.

        Parameters
        ----------
        trust_scores : np.ndarray
            Trust scores.
        correct : np.ndarray
            Binary correctness labels.
        trust_threshold : float
            Threshold for HIGH vs LOW trust.

        Returns
        -------
        dict
            Contingency table, rates, and analysis.
        """
        high_trust = trust_scores >= trust_threshold
        low_trust = ~high_trust

        # Contingency table
        ht_correct = int((high_trust & (correct == 1)).sum())
        ht_incorrect = int((high_trust & (correct == 0)).sum())
        lt_correct = int((low_trust & (correct == 1)).sum())
        lt_incorrect = int((low_trust & (correct == 0)).sum())

        n = len(trust_scores)

        result = {
            "trust_threshold": trust_threshold,
            "contingency_table": {
                "high_trust_correct": ht_correct,
                "high_trust_incorrect": ht_incorrect,
                "low_trust_correct": lt_correct,
                "low_trust_incorrect": lt_incorrect,
            },
            "rates": {
                "high_trust_accuracy": round(
                    ht_correct / (ht_correct + ht_incorrect), 4
                ) if (ht_correct + ht_incorrect) > 0 else None,
                "low_trust_accuracy": round(
                    lt_correct / (lt_correct + lt_incorrect), 4
                ) if (lt_correct + lt_incorrect) > 0 else None,
                "dangerous_rate": round(ht_incorrect / n, 4),
                "lucky_rate": round(lt_correct / n, 4),
            },
            "n_samples": n,
        }

        logger.info("Trust-Correctness Decorrelation:")
        for k, v in result["contingency_table"].items():
            logger.info(f"  {k}: {v}")
        logger.info(f"  Dangerous rate (HT + wrong): {result['rates']['dangerous_rate']}")
        logger.info(f"  Lucky rate (LT + correct): {result['rates']['lucky_rate']}")

        return result

    # ── Combined evaluation ──────────────────────────────────────────────

    def evaluate_all(
        self,
        predictions: list[str],
        gold_labels: list[str],
        trust_scores: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run all evaluation metrics.

        Returns
        -------
        dict
            Combined correctness, calibration, discrimination, and
            decorrelation results.
        """
        correct_arr = np.array(
            [1 if p.lower() == g.lower() else 0 for p, g in zip(predictions, gold_labels)]
        )

        results = {
            "correctness": self.compute_correctness(predictions, gold_labels),
        }

        if trust_scores is not None:
            results["calibration"] = self.compute_calibration(trust_scores, correct_arr)
            results["discrimination"] = self.compute_discrimination(
                trust_scores, correct_arr
            )
            results["decorrelation"] = self.compute_decorrelation(
                trust_scores, correct_arr
            )

        return results
