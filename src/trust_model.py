"""
TRUST-RAG V2 — Trust Model
============================
Calibrated logistic regression trust estimator.

Supports three architectures:
  A. Heuristic weighted score (baseline)
  B. Calibrated logistic regression (recommended)
  C. Gaussian Naive Bayes (Bayesian alternative)

Usage
-----
    from src.trust_model import TrustModel
    model = TrustModel(model_type="logistic_regression")
    model.fit(X_train, y_train, X_val, y_val)
    scores = model.predict_trust(X_test)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from src.config import get_default_config
from src.trust_features import FEATURE_NAMES
from src.utils import get_logger, save_json

logger = get_logger(__name__)


class TrustModel:
    """Trust estimation model with calibration support."""

    def __init__(
        self,
        model_type: str = "logistic_regression",
        regularization_C: float = 1.0,
        cv_folds: int = 5,
        calibration_method: str = "platt",  # platt | isotonic
        trust_levels: dict[str, float] | None = None,
        seed: int = 42,
    ):
        self.model_type = model_type
        self.regularization_C = regularization_C
        self.cv_folds = cv_folds
        self.calibration_method = calibration_method
        self.trust_levels = trust_levels or {"high": 0.70, "moderate": 0.40}
        self.seed = seed

        self._model = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._feature_names = FEATURE_NAMES

    @classmethod
    def from_config(cls, config: dict | None = None) -> "TrustModel":
        if config is None:
            config = get_default_config()
        trust_cfg = config.get("trust", {})
        return cls(
            model_type=trust_cfg.get("model_type", "logistic_regression"),
            regularization_C=trust_cfg.get("regularization_C", 1.0),
            cv_folds=trust_cfg.get("cv_folds", 5),
            calibration_method=trust_cfg.get("calibration_method", "platt"),
            trust_levels=trust_cfg.get("trust_levels"),
            seed=config.get("seed", 42),
        )

    # ── Heuristic trust (Architecture A) ─────────────────────────────────

    def _heuristic_score(self, X: np.ndarray) -> np.ndarray:
        """Compute heuristic trust score with equal weights.

        Uses 4 key features with equal 0.25 weighting:
          retrieval_top1_score, mean_max_entailment,
          (1 - max_contradiction), answer_evidence_sim
        """
        # Feature indices
        idx_ret = FEATURE_NAMES.index("retrieval_top1_score")
        idx_ent = FEATURE_NAMES.index("mean_max_entailment")
        idx_con = FEATURE_NAMES.index("max_contradiction")
        idx_sim = FEATURE_NAMES.index("answer_evidence_sim")

        scores = (
            0.25 * X[:, idx_ret]
            + 0.25 * X[:, idx_ent]
            + 0.25 * (1.0 - X[:, idx_con])
            + 0.25 * X[:, idx_sim]
        )
        return np.clip(scores, 0.0, 1.0)

    # ── Fitting ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit the trust model.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_train, 9).
        y_train : np.ndarray
            Training labels, shape (n_train,). 1=correct, 0=incorrect.
        X_val : np.ndarray | None
            Validation features for calibration.
        y_val : np.ndarray | None
            Validation labels for calibration.

        Returns
        -------
        dict
            Training metrics including CV scores and feature coefficients.
        """
        if self.model_type == "heuristic":
            logger.info("Heuristic trust model — no training needed")
            self._is_fitted = True
            return {"model_type": "heuristic", "cv_scores": None}

        # Scale features
        X_train_scaled = self._scaler.fit_transform(X_train)

        if self.model_type == "logistic_regression":
            base_model = LogisticRegression(
                C=self.regularization_C,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.seed,
            )
        elif self.model_type == "bayesian":
            base_model = GaussianNB()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Cross-validation on train set
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.seed
        )
        cv_scores = cross_val_score(
            base_model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
        )
        logger.info(
            f"CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # Fit on full train set
        base_model.fit(X_train_scaled, y_train)

        # Calibrate on validation set if available
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scaler.transform(X_val)
            method = (
                "sigmoid" if self.calibration_method == "platt" else "isotonic"
            )
            calibrated = CalibratedClassifierCV(
                base_model, method=method, cv="prefit"
            )
            calibrated.fit(X_val_scaled, y_val)
            self._model = calibrated
            logger.info(f"Calibrated with {method} method on {len(y_val)} examples")
        else:
            self._model = base_model
            logger.warning("No validation set — model is NOT calibrated")

        self._is_fitted = True

        # Extract feature coefficients (logistic regression only)
        metrics: dict[str, Any] = {
            "model_type": self.model_type,
            "cv_auroc_mean": round(float(cv_scores.mean()), 4),
            "cv_auroc_std": round(float(cv_scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in cv_scores],
        }

        if self.model_type == "logistic_regression" and hasattr(
            base_model, "coef_"
        ):
            coefficients = dict(
                zip(self._feature_names, base_model.coef_[0].tolist())
            )
            metrics["coefficients"] = {
                k: round(v, 4) for k, v in coefficients.items()
            }
            metrics["intercept"] = round(float(base_model.intercept_[0]), 4)

            # Log feature importance
            logger.info("Feature coefficients (scaled):")
            sorted_coefs = sorted(
                coefficients.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for name, coef in sorted_coefs:
                logger.info(f"  {name}: {coef:.4f}")

        return metrics

    # ── Prediction ───────────────────────────────────────────────────────

    def predict_trust(self, X: np.ndarray) -> np.ndarray:
        """Predict trust scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, 9).

        Returns
        -------
        np.ndarray
            Trust scores in [0, 1], shape (n_samples,).
        """
        if self.model_type == "heuristic":
            return self._heuristic_score(X)

        if not self._is_fitted:
            raise RuntimeError("Trust model not fitted. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        # predict_proba returns P(class=0), P(class=1)
        probs = self._model.predict_proba(X_scaled)
        return probs[:, 1]  # P(correct)

    def predict_trust_level(self, score: float) -> str:
        """Classify trust score into HIGH/MODERATE/LOW."""
        if score >= self.trust_levels["high"]:
            return "HIGH"
        elif score >= self.trust_levels["moderate"]:
            return "MODERATE"
        else:
            return "LOW"

    def predict_batch(
        self, X: np.ndarray
    ) -> list[dict[str, Any]]:
        """Predict trust scores and levels for a batch.

        Returns
        -------
        list[dict]
            Each dict has "trust_score" and "trust_level".
        """
        scores = self.predict_trust(X)
        results = []
        for score in scores:
            results.append({
                "trust_score": round(float(score), 4),
                "trust_level": self.predict_trust_level(score),
            })
        return results

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the trust model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "model_type": self.model_type,
            "trust_levels": self.trust_levels,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Trust model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TrustModel":
        """Load a trust model from disk."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        model = cls(model_type=data["model_type"], trust_levels=data["trust_levels"])
        model._model = data["model"]
        model._scaler = data["scaler"]
        model._feature_names = data["feature_names"]
        model._is_fitted = data["is_fitted"]
        return model
