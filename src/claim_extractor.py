"""
TRUST-RAG V2 — Claim Extractor
================================
Decomposes LLM-generated answers into atomic claims for NLI verification.

Strategy: SciSpaCy sentence splitting + heuristic filtering.
Falls back to regex sentence splitting if SciSpaCy is unavailable.

Usage
-----
    from src.claim_extractor import ClaimExtractor
    extractor = ClaimExtractor()
    claims = extractor.extract("RPS detected 39 of 107 cases. This suggests...")
"""

from __future__ import annotations

import re
from typing import Any

from src.config import get_default_config
from src.utils import get_logger

logger = get_logger(__name__)


class ClaimExtractor:
    """Extract verifiable claims from generated text."""

    # Prefixes that indicate meta-statements (keep them — they often
    # contain the actual conclusion)
    META_PREFIXES = [
        "based on",
        "according to",
        "in summary",
        "in conclusion",
        "overall",
        "therefore",
        "thus",
        "hence",
        "consequently",
    ]

    def __init__(
        self,
        spacy_model: str = "en_core_sci_sm",
        min_claim_tokens: int = 5,
        meta_prefixes: list[str] | None = None,
    ):
        self.spacy_model_name = spacy_model
        self.min_claim_tokens = min_claim_tokens
        self.meta_prefixes = meta_prefixes or self.META_PREFIXES
        self._nlp = None
        self._use_spacy = True

    @classmethod
    def from_config(cls, config: dict | None = None) -> "ClaimExtractor":
        if config is None:
            config = get_default_config()
        claim_cfg = config.get("claim_extraction", {})
        return cls(
            spacy_model=claim_cfg.get("spacy_model", "en_core_sci_sm"),
            min_claim_tokens=claim_cfg.get("min_claim_tokens", 5),
            meta_prefixes=claim_cfg.get("meta_prefixes"),
        )

    # ── spaCy loading ────────────────────────────────────────────────────

    def _load_spacy(self) -> None:
        """Load SciSpaCy model, falling back to en_core_web_sm or regex if unavailable."""
        if self._nlp is not None:
            return

        try:
            import spacy

            try:
                self._nlp = spacy.load(self.spacy_model_name)
                logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
            except OSError:
                logger.info(f"'{self.spacy_model_name}' not found; trying 'en_core_web_sm'...")
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
        except (ImportError, OSError) as e:
            logger.warning(
                f"SpaCy models not available: {e}. "
                f"Falling back to regex sentence splitting."
            )
            self._use_spacy = False

    # ── Sentence splitting ───────────────────────────────────────────────

    def _split_spacy(self, text: str) -> list[str]:
        """Split text into sentences using SciSpaCy."""
        self._load_spacy()
        if not self._use_spacy:
            return self._split_regex(text)

        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def _split_regex(self, text: str) -> list[str]:
        """Fallback regex sentence splitter."""
        # Split on period/question-mark/exclamation followed by space + capital
        # or end of string. Handles common abbreviations.
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z])',
            text.strip(),
        )
        # Also split on newlines
        expanded = []
        for s in sentences:
            expanded.extend(s.split("\n"))
        return [s.strip() for s in expanded if s.strip()]

    # ── Filtering ────────────────────────────────────────────────────────

    def _is_valid_claim(self, sentence: str) -> bool:
        """Check if a sentence qualifies as a verifiable claim."""
        tokens = sentence.split()

        # Too short
        if len(tokens) < self.min_claim_tokens:
            return False

        # Empty or whitespace
        if not sentence.strip():
            return False

        # Skip pure questions
        if sentence.strip().endswith("?"):
            return False

        # Skip list markers alone
        if re.match(r"^\s*[\-\*\d]+[.)]\s*$", sentence):
            return False

        return True

    def _is_meta_statement(self, sentence: str) -> bool:
        """Check if sentence starts with a meta-prefix."""
        lower = sentence.lower().strip()
        return any(lower.startswith(p) for p in self.meta_prefixes)

    # ── Main extraction ──────────────────────────────────────────────────

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract claims from generated text.

        Parameters
        ----------
        text : str
            LLM-generated answer text.

        Returns
        -------
        list[dict]
            List of claim dicts:
            {
                "claim": str,
                "claim_index": int,
                "is_meta": bool,
                "n_tokens": int,
            }
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._split_spacy(text) if self._use_spacy else self._split_regex(text)

        claims = []
        idx = 0
        for sent in sentences:
            sent = sent.strip()
            if not self._is_valid_claim(sent):
                continue

            claims.append({
                "claim": sent,
                "claim_index": idx,
                "is_meta": self._is_meta_statement(sent),
                "n_tokens": len(sent.split()),
            })
            idx += 1

        return claims

    def extract_batch(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """Extract claims from a batch of texts."""
        results = []
        total_claims = 0
        for text in texts:
            claims = self.extract(text)
            results.append(claims)
            total_claims += len(claims)

        avg_claims = total_claims / len(texts) if texts else 0
        logger.info(
            f"Claim extraction: {total_claims} claims from {len(texts)} texts "
            f"(avg {avg_claims:.1f} claims/text)"
        )
        return results
