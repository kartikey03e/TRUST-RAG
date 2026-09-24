"""
TRUST-RAG V2 — Decision Extractor
===================================
Extracts a yes/no/maybe decision from LLM-generated free-form text.

Strategy:
  1. Regex pattern matching (fast, ~85% reliable)
  2. Zero-shot classification fallback (slower, ~90% reliable)

Usage
-----
    from src.decision_extractor import DecisionExtractor
    extractor = DecisionExtractor.from_config(config)
    decision = extractor.extract("The evidence suggests that... Therefore, YES.")
"""

from __future__ import annotations

import re
from typing import Any

from src.config import get_default_config
from src.utils import clear_gpu_memory, get_logger

logger = get_logger(__name__)

VALID_DECISIONS = {"yes", "no", "maybe"}


class DecisionExtractor:
    """Extract yes/no/maybe decision from generated text."""

    # Regex patterns, ordered from most specific to least
    PATTERNS = [
        # Explicit decision markers
        r"(?:final\s+)?(?:answer|decision|conclusion|verdict)\s*(?:is|:)\s*(yes|no|maybe)",
        # "the answer is yes/no/maybe"
        r"the\s+answer\s+(?:is|to\s+this\s+question\s+is)\s+[\"']?(yes|no|maybe)[\"']?",
        # Standalone on last line
        r"^\s*(yes|no|maybe)\s*[.!]?\s*$",
        # "YES/NO/MAYBE" at end of text
        r"\b(yes|no|maybe)\s*[.!]?\s*$",
    ]

    def __init__(
        self,
        fallback_model: str = "facebook/bart-large-mnli",
        candidate_labels: list[str] | None = None,
    ):
        self.fallback_model = fallback_model
        self.candidate_labels = candidate_labels or ["yes", "no", "maybe"]
        self._classifier = None

    @classmethod
    def from_config(cls, config: dict | None = None) -> "DecisionExtractor":
        if config is None:
            config = get_default_config()
        dec_cfg = config.get("decision_extraction", {})
        return cls(
            fallback_model=dec_cfg.get("fallback_model", "facebook/bart-large-mnli"),
            candidate_labels=dec_cfg.get("candidate_labels", ["yes", "no", "maybe"]),
        )

    # ── Regex extraction ─────────────────────────────────────────────────

    def _extract_regex(self, text: str) -> str | None:
        """Try to extract decision using regex patterns."""
        text_lower = text.lower().strip()

        for pattern in self.PATTERNS:
            match = re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE)
            if match:
                decision = match.group(1).lower()
                if decision in VALID_DECISIONS:
                    return decision

        return None

    # ── Zero-shot fallback ───────────────────────────────────────────────

    def _load_classifier(self) -> None:
        """Lazy-load the zero-shot classification pipeline."""
        if self._classifier is not None:
            return

        from transformers import pipeline

        logger.info(f"Loading zero-shot classifier: {self.fallback_model}")
        self._classifier = pipeline(
            "zero-shot-classification",
            model=self.fallback_model,
            device=-1,  # CPU to save GPU memory
        )

    def _extract_zero_shot(self, text: str) -> str:
        """Extract decision using zero-shot classification."""
        self._load_classifier()

        # Use the last ~200 chars (most likely to contain the conclusion)
        text_tail = text[-500:] if len(text) > 500 else text

        result = self._classifier(
            text_tail,
            candidate_labels=self.candidate_labels,
            hypothesis_template="The answer to the question is {}.",
        )
        return result["labels"][0].lower()

    def unload_classifier(self) -> None:
        """Free classifier memory."""
        del self._classifier
        self._classifier = None
        clear_gpu_memory()

    # ── Main extraction ──────────────────────────────────────────────────

    def extract(self, text: str) -> dict[str, Any]:
        """Extract a decision from generated text.

        Returns
        -------
        dict
            {
                "decision": "yes" | "no" | "maybe",
                "method": "regex" | "zero_shot",
                "confidence": float | None,
            }
        """
        if not text or not text.strip():
            return {"decision": "maybe", "method": "empty_input", "confidence": None}

        # Stage 1: Regex
        decision = self._extract_regex(text)
        if decision:
            return {"decision": decision, "method": "regex", "confidence": None}

        # Stage 2: Zero-shot fallback
        logger.debug("Regex extraction failed, falling back to zero-shot")
        decision = self._extract_zero_shot(text)
        return {"decision": decision, "method": "zero_shot", "confidence": None}

    def extract_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Extract decisions from a batch of generated texts."""
        results = []
        regex_count = 0
        zs_count = 0

        for text in texts:
            r = self.extract(text)
            results.append(r)
            if r["method"] == "regex":
                regex_count += 1
            else:
                zs_count += 1

        logger.info(
            f"Decision extraction: {regex_count} regex, {zs_count} zero-shot "
            f"({regex_count / len(texts) * 100:.1f}% regex success)"
        )
        return results
