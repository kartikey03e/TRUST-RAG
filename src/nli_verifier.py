"""
TRUST-RAG V2 — NLI Verifier
==============================
Performs claim-level evidence verification using a cross-encoder NLI model.

For each (claim, evidence) pair, produces:
  P(entailment), P(contradiction), P(neutral)

Then aggregates per-claim and per-answer scores.

Usage
-----
    from src.nli_verifier import NLIVerifier
    verifier = NLIVerifier.from_config(config)
    result = verifier.verify_claim("RPS detected 39 cases", "evidence text...")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.config import get_default_config
from src.utils import clear_gpu_memory, get_logger, print_gpu_memory, timer

logger = get_logger(__name__)


@dataclass
class ClaimVerificationResult:
    """NLI result for a single claim against one or more evidence passages."""

    claim: str
    claim_index: int
    # Per-evidence scores (list of dicts, one per evidence passage)
    per_evidence: list[dict[str, float]] = field(default_factory=list)
    # Aggregated scores (max across evidence passages)
    max_entailment: float = 0.0
    max_contradiction: float = 0.0
    max_neutral: float = 0.0
    # Best-supporting evidence index
    best_evidence_index: int = -1

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "claim_index": self.claim_index,
            "max_entailment": round(self.max_entailment, 4),
            "max_contradiction": round(self.max_contradiction, 4),
            "max_neutral": round(self.max_neutral, 4),
            "best_evidence_index": self.best_evidence_index,
            "n_evidence_passages": len(self.per_evidence),
        }


@dataclass
class AnswerVerificationResult:
    """Aggregated NLI verification for an entire generated answer."""

    question_pmid: str
    n_claims: int
    claim_results: list[ClaimVerificationResult]
    # Aggregated answer-level metrics
    mean_max_entailment: float = 0.0
    frac_entailed: float = 0.0       # fraction with entailment > threshold
    max_contradiction: float = 0.0
    frac_contradicted: float = 0.0   # fraction with contradiction > threshold
    unsupported_rate: float = 0.0     # fraction with entailment < threshold

    def to_dict(self) -> dict:
        return {
            "question_pmid": self.question_pmid,
            "n_claims": self.n_claims,
            "mean_max_entailment": round(self.mean_max_entailment, 4),
            "frac_entailed": round(self.frac_entailed, 4),
            "max_contradiction": round(self.max_contradiction, 4),
            "frac_contradicted": round(self.frac_contradicted, 4),
            "unsupported_rate": round(self.unsupported_rate, 4),
            "claim_results": [c.to_dict() for c in self.claim_results],
        }


class NLIVerifier:
    """Cross-encoder NLI verifier for claim-level evidence verification."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        batch_size: int = 32,
        label_map: dict[int, str] | None = None,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.label_map = label_map or {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.device = device

        self._model = None
        self._tokenizer = None

        # Reverse map to get index from label
        self._label_to_idx = {v: k for k, v in self.label_map.items()}

    @classmethod
    def from_config(cls, config: dict | None = None) -> "NLIVerifier":
        if config is None:
            config = get_default_config()
        nli_cfg = config.get("nli", {})
        from src.config import get_device

        return cls(
            model_name=nli_cfg.get("model", "cross-encoder/nli-deberta-v3-base"),
            batch_size=nli_cfg.get("batch_size", 32),
            label_map={int(k): v for k, v in nli_cfg.get("label_map", {}).items()},
            entailment_threshold=nli_cfg.get("thresholds", {}).get("entailment", 0.5),
            contradiction_threshold=nli_cfg.get("thresholds", {}).get("contradiction", 0.5),
            device=get_device(),
        )

    # ── Model loading ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the cross-encoder NLI model."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading NLI model: {self.model_name}")
        print_gpu_memory("before NLI load")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        print_gpu_memory("after NLI load")

    def unload_model(self) -> None:
        """Free NLI model from memory."""
        del self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        clear_gpu_memory()
        logger.info("NLI model unloaded")

    # ── Core NLI scoring ─────────────────────────────────────────────────

    def _score_pairs(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[dict[str, float]]:
        """Score (premise, hypothesis) pairs using the NLI model.

        Parameters
        ----------
        premises : list[str]
            Evidence texts.
        hypotheses : list[str]
            Claim texts.

        Returns
        -------
        list[dict]
            List of {"entailment": p, "contradiction": p, "neutral": p}.
        """
        if self._model is None:
            self.load_model()

        all_scores = []

        for start in range(0, len(premises), self.batch_size):
            batch_premises = premises[start : start + self.batch_size]
            batch_hypotheses = hypotheses[start : start + self.batch_size]

            encoded = self._tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encoded).logits
                probs = F.softmax(logits, dim=1).cpu().numpy()

            for prob_row in probs:
                scores = {}
                for idx, label in self.label_map.items():
                    scores[label] = float(prob_row[idx])
                all_scores.append(scores)

        return all_scores

    # ── Claim-level verification ─────────────────────────────────────────

    def verify_claim(
        self,
        claim: str,
        evidence_passages: list[str],
        claim_index: int = 0,
    ) -> ClaimVerificationResult:
        """Verify a single claim against multiple evidence passages.

        Parameters
        ----------
        claim : str
            The claim to verify.
        evidence_passages : list[str]
            Evidence texts to check against.
        claim_index : int
            Index of this claim in the answer.

        Returns
        -------
        ClaimVerificationResult
        """
        if not evidence_passages:
            return ClaimVerificationResult(
                claim=claim, claim_index=claim_index,
            )

        # Score claim against each evidence passage
        premises = evidence_passages
        hypotheses = [claim] * len(evidence_passages)
        scores = self._score_pairs(premises, hypotheses)

        # Find max entailment/contradiction across passages
        entailments = [s["entailment"] for s in scores]
        contradictions = [s["contradiction"] for s in scores]
        neutrals = [s["neutral"] for s in scores]

        best_idx = int(np.argmax(entailments))

        return ClaimVerificationResult(
            claim=claim,
            claim_index=claim_index,
            per_evidence=scores,
            max_entailment=max(entailments),
            max_contradiction=max(contradictions),
            max_neutral=max(neutrals),
            best_evidence_index=best_idx,
        )

    # ── Answer-level verification ────────────────────────────────────────

    def verify_answer(
        self,
        claims: list[dict],
        evidence_passages: list[str],
        question_pmid: str = "",
    ) -> AnswerVerificationResult:
        """Verify all claims in an answer against evidence.

        Parameters
        ----------
        claims : list[dict]
            Claim dicts from ClaimExtractor (must have "claim" key).
        evidence_passages : list[str]
            Evidence texts.
        question_pmid : str
            For record-keeping.

        Returns
        -------
        AnswerVerificationResult
        """
        if not claims:
            return AnswerVerificationResult(
                question_pmid=question_pmid,
                n_claims=0,
                claim_results=[],
            )

        claim_results = []
        for claim_dict in claims:
            cr = self.verify_claim(
                claim=claim_dict["claim"],
                evidence_passages=evidence_passages,
                claim_index=claim_dict.get("claim_index", 0),
            )
            claim_results.append(cr)

        # Aggregate
        entailments = [cr.max_entailment for cr in claim_results]
        contradictions = [cr.max_contradiction for cr in claim_results]

        n = len(claim_results)
        mean_ent = float(np.mean(entailments))
        frac_ent = sum(1 for e in entailments if e > self.entailment_threshold) / n
        max_contra = max(contradictions)
        frac_contra = sum(
            1 for c in contradictions if c > self.contradiction_threshold
        ) / n
        unsup = sum(1 for e in entailments if e < self.entailment_threshold) / n

        return AnswerVerificationResult(
            question_pmid=question_pmid,
            n_claims=n,
            claim_results=claim_results,
            mean_max_entailment=mean_ent,
            frac_entailed=frac_ent,
            max_contradiction=max_contra,
            frac_contradicted=frac_contra,
            unsupported_rate=unsup,
        )
