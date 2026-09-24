"""
TRUST-RAG V2 — Error Analysis
================================
Classifies incorrect answers into a 6-category error taxonomy:

  E1: Retrieval Failure       — gold PMID not in top-5
  E2: Ranking Failure         — gold PMID in top-5 but not rank 1
  E3: Evidence Ambiguity      — evidence contradicts gold answer
  E4: Decision Error          — reasoning entailed but wrong classification
  E5: Hallucination           — claims contradict evidence
  E6: Misinterpretation       — claims neither entailed nor contradicted

Usage
-----
    from src.error_analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer()
    errors = analyzer.classify_errors(pipeline_results)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

ERROR_CATEGORIES = {
    "E1_RETRIEVAL_FAILURE": "Gold PMID not retrieved in top-5",
    "E2_RANKING_FAILURE": "Gold PMID retrieved but not rank 1",
    "E3_EVIDENCE_AMBIGUITY": "Evidence contradicts the gold answer",
    "E4_DECISION_ERROR": "Reasoning is sound but decision is wrong",
    "E5_HALLUCINATION": "Claims contradict the evidence",
    "E6_MISINTERPRETATION": "Claims neither entailed nor contradicted",
}


class ErrorAnalyzer:
    """Classify incorrect answers into error categories."""

    def __init__(
        self,
        entailment_threshold: float = 0.6,
        contradiction_threshold: float = 0.6,
    ):
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def classify_single(self, result: dict[str, Any]) -> dict[str, Any]:
        """Classify a single incorrect answer.

        Parameters
        ----------
        result : dict
            Pipeline result with keys:
            - correct: bool
            - gold_pmid_in_top5: bool
            - gold_pmid_rank: int (0 if not found)
            - mean_max_entailment: float
            - max_contradiction: float
            - predicted_decision: str
            - gold_decision: str

        Returns
        -------
        dict
            Error classification.
        """
        if result.get("correct", True):
            return {
                "error_type": "CORRECT",
                "description": "Answer is correct",
            }

        # E1: Retrieval Failure
        if not result.get("gold_pmid_in_top5", True):
            return {
                "error_type": "E1_RETRIEVAL_FAILURE",
                "description": ERROR_CATEGORIES["E1_RETRIEVAL_FAILURE"],
                "gold_pmid_rank": result.get("gold_pmid_rank", 0),
            }

        # E2: Ranking Failure
        gold_rank = result.get("gold_pmid_rank", 0)
        if gold_rank > 1:
            return {
                "error_type": "E2_RANKING_FAILURE",
                "description": ERROR_CATEGORIES["E2_RANKING_FAILURE"],
                "gold_pmid_rank": gold_rank,
            }

        # Now we know gold PMID is rank 1 — failure is in generation/verification
        mean_ent = result.get("mean_max_entailment", 0.0)
        max_contra = result.get("max_contradiction", 0.0)

        # E5: Hallucination — claims contradict evidence
        if max_contra > self.contradiction_threshold:
            return {
                "error_type": "E5_HALLUCINATION",
                "description": ERROR_CATEGORIES["E5_HALLUCINATION"],
                "max_contradiction": max_contra,
            }

        # E4: Decision Error — reasoning seems supported but decision is wrong
        if mean_ent > self.entailment_threshold:
            return {
                "error_type": "E4_DECISION_ERROR",
                "description": ERROR_CATEGORIES["E4_DECISION_ERROR"],
                "mean_entailment": mean_ent,
                "predicted": result.get("predicted_decision", ""),
                "gold": result.get("gold_decision", ""),
            }

        # E3: Evidence Ambiguity
        # Check if the evidence itself is ambiguous (moderate contradiction + moderate entailment)
        if max_contra > 0.3 and mean_ent > 0.3:
            return {
                "error_type": "E3_EVIDENCE_AMBIGUITY",
                "description": ERROR_CATEGORIES["E3_EVIDENCE_AMBIGUITY"],
                "max_contradiction": max_contra,
                "mean_entailment": mean_ent,
            }

        # E6: Misinterpretation — claims not supported by evidence
        return {
            "error_type": "E6_MISINTERPRETATION",
            "description": ERROR_CATEGORIES["E6_MISINTERPRETATION"],
            "mean_entailment": mean_ent,
            "max_contradiction": max_contra,
        }

    def classify_errors(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Classify all incorrect answers in a result set.

        Parameters
        ----------
        results : list[dict]
            List of pipeline results.

        Returns
        -------
        dict
            Error taxonomy with counts, rates, and per-result classifications.
        """
        classifications = []
        for r in results:
            cls_result = self.classify_single(r)
            cls_result["question_pmid"] = r.get("question_pmid", "")
            cls_result["question"] = r.get("question", "")
            classifications.append(cls_result)

        # Count error types
        error_types = [c["error_type"] for c in classifications]
        counts = dict(Counter(error_types))

        n_total = len(results)
        n_incorrect = sum(1 for c in classifications if c["error_type"] != "CORRECT")

        # Rates
        rates = {
            k: round(v / n_incorrect, 4) if n_incorrect > 0 else 0.0
            for k, v in counts.items()
            if k != "CORRECT"
        }

        # Per gold-decision breakdown
        gold_error_dist: dict[str, dict[str, int]] = {}
        for c, r in zip(classifications, results):
            gold = r.get("gold_decision", "unknown")
            et = c["error_type"]
            if et == "CORRECT":
                continue
            if gold not in gold_error_dist:
                gold_error_dist[gold] = {}
            gold_error_dist[gold][et] = gold_error_dist[gold].get(et, 0) + 1

        summary = {
            "n_total": n_total,
            "n_correct": counts.get("CORRECT", 0),
            "n_incorrect": n_incorrect,
            "error_counts": {k: v for k, v in counts.items() if k != "CORRECT"},
            "error_rates": rates,
            "gold_decision_error_distribution": gold_error_dist,
            "classifications": classifications,
        }

        logger.info("=" * 50)
        logger.info("ERROR ANALYSIS")
        logger.info("=" * 50)
        logger.info(f"Total: {n_total} | Correct: {counts.get('CORRECT', 0)} | Incorrect: {n_incorrect}")
        for et, count in sorted(counts.items()):
            if et != "CORRECT":
                logger.info(f"  {et}: {count} ({rates.get(et, 0):.1%})")

        return summary
