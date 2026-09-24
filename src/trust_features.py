"""
TRUST-RAG V2 — Trust Feature Constructor
==========================================
Builds the 9-dimensional trust feature vector from retrieval, NLI,
and semantic similarity signals.

Features (9):
  Retrieval (3): top1_score, score_margin, n_unique_pmids
  NLI (4): mean_max_entailment, frac_entailed, max_contradiction, frac_contradicted
  Semantic (2): answer_evidence_sim, question_answer_sim

Usage
-----
    from src.trust_features import TrustFeatureBuilder
    builder = TrustFeatureBuilder()
    features = builder.build(retrieval_result, nli_result, answer, evidence, question)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "retrieval_top1_score",
    "retrieval_score_margin",
    "n_unique_pmids",
    "mean_max_entailment",
    "frac_entailed_claims",
    "max_contradiction",
    "frac_contradicted_claims",
    "answer_evidence_sim",
    "question_answer_sim",
]


@dataclass
class TrustFeatures:
    """Container for trust feature vector."""

    question_pmid: str
    features: dict[str, float]
    feature_vector: np.ndarray  # shape (9,)

    def to_dict(self) -> dict:
        result = {"question_pmid": self.question_pmid}
        result.update({k: round(v, 6) for k, v in self.features.items()})
        return result


class TrustFeatureBuilder:
    """Constructs trust feature vectors from pipeline outputs."""

    def __init__(self, embedding_model_name: str | None = None):
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding for semantic similarity computation."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            model_name = self._embedding_model_name or "all-MiniLM-L6-v2"
            self._embedding_model = SentenceTransformer(model_name, device="cpu")
            logger.info(f"Loaded embedding model: {model_name}")

        return self._embedding_model.encode(text, normalize_embeddings=True)

    def _cosine_sim(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        emb_a = self._get_embedding(text_a)
        emb_b = self._get_embedding(text_b)
        return float(np.dot(emb_a, emb_b))

    def build(
        self,
        retrieval_result: dict[str, Any],
        nli_result: dict[str, Any],
        generated_answer: str,
        evidence_text: str,
        question: str,
        question_pmid: str = "",
    ) -> TrustFeatures:
        """Build the 9-dimensional trust feature vector.

        Parameters
        ----------
        retrieval_result : dict
            Must contain: top1_score, score_margin, retrieved_pmids
        nli_result : dict
            Must contain: mean_max_entailment, frac_entailed,
                          max_contradiction, frac_contradicted
        generated_answer : str
            The LLM's generated answer text.
        evidence_text : str
            Concatenated evidence text.
        question : str
            The original question.
        question_pmid : str
            For record-keeping.

        Returns
        -------
        TrustFeatures
        """
        # ── Retrieval features (3) ───────────────────────────────────
        top1_score = float(retrieval_result.get("top1_score", 0.0))
        score_margin = float(retrieval_result.get("score_margin", 0.0))
        n_pmids = len(set(retrieval_result.get("retrieved_pmids", [])))

        # Normalize n_pmids to [0, 1] range (max 5 unique PMIDs)
        n_pmids_norm = min(n_pmids / 5.0, 1.0)

        # ── NLI features (4) ────────────────────────────────────────
        mean_ent = float(nli_result.get("mean_max_entailment", 0.0))
        frac_ent = float(nli_result.get("frac_entailed", 0.0))
        max_contra = float(nli_result.get("max_contradiction", 0.0))
        frac_contra = float(nli_result.get("frac_contradicted", 0.0))

        # ── Semantic features (2) ───────────────────────────────────
        ans_evi_sim = self._cosine_sim(generated_answer, evidence_text)
        q_ans_sim = self._cosine_sim(question, generated_answer)

        # ── Assemble ────────────────────────────────────────────────
        features = {
            "retrieval_top1_score": top1_score,
            "retrieval_score_margin": score_margin,
            "n_unique_pmids": n_pmids_norm,
            "mean_max_entailment": mean_ent,
            "frac_entailed_claims": frac_ent,
            "max_contradiction": max_contra,
            "frac_contradicted_claims": frac_contra,
            "answer_evidence_sim": ans_evi_sim,
            "question_answer_sim": q_ans_sim,
        }

        feature_vector = np.array(
            [features[name] for name in FEATURE_NAMES], dtype=np.float32
        )

        return TrustFeatures(
            question_pmid=question_pmid,
            features=features,
            feature_vector=feature_vector,
        )

    def build_batch(
        self,
        retrieval_results: list[dict],
        nli_results: list[dict],
        generated_answers: list[str],
        evidence_texts: list[str],
        questions: list[str],
        question_pmids: list[str],
    ) -> list[TrustFeatures]:
        """Build feature vectors for a batch of questions."""
        results = []
        for i in range(len(questions)):
            tf = self.build(
                retrieval_result=retrieval_results[i],
                nli_result=nli_results[i],
                generated_answer=generated_answers[i],
                evidence_text=evidence_texts[i],
                question=questions[i],
                question_pmid=question_pmids[i],
            )
            results.append(tf)
        logger.info(f"Built {len(results)} trust feature vectors")
        return results

    def get_feature_matrix(
        self, trust_features_list: list[TrustFeatures]
    ) -> np.ndarray:
        """Stack feature vectors into a matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 9).
        """
        return np.vstack([tf.feature_vector for tf in trust_features_list])
