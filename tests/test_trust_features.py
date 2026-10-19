"""
Unit tests for TrustFeatureBuilder and n_unique_pmids retrieval diversity feature.
Verifies Phase 2 requirements:
- five candidates containing five unique PMIDs (norm = 1.0)
- five candidates containing duplicate PMIDs (e.g. 3 unique, norm = 0.6)
- fewer than five available candidates (e.g. 2 candidates, norm = 0.4)
- an empty candidate list (norm = 0.0)
- duplicate chunks from one PMID (e.g. 5 chunks all from 1 PMID, norm = 0.2)
- correct separation of candidate-pool and generator-evidence inputs
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.trust_features import TrustFeatureBuilder, FEATURE_NAMES


class TestTrustFeatures(unittest.TestCase):
    def setUp(self):
        # Initialize builder with mock embedding method to avoid loading neural network in unit tests
        self.builder = TrustFeatureBuilder()
        self.builder._get_embedding = MagicMock(return_value=np.ones(384, dtype=np.float32) / np.sqrt(384))

        self.mock_nli = {
            "mean_max_entailment": 0.85,
            "frac_entailed": 0.75,
            "max_contradiction": 0.10,
            "frac_contradicted": 0.05,
        }
        self.question = "Does medication X improve condition Y?"
        self.answer = "Medication X significantly improves condition Y in clinical trials."
        self.evidence = "Clinical trials show medication X improves condition Y."

    def test_five_candidates_five_unique_pmids(self):
        """Case 1: Five candidates containing five unique PMIDs should yield 5/5 = 1.0."""
        ret_result = {
            "top1_score": 0.035,
            "score_margin": 0.010,
            "candidate_pmids": ["10001", "10002", "10003", "10004", "10005"],
            "evidence_pmids": ["10001", "10002", "10003"],
        }
        tf = self.builder.build(
            retrieval_result=ret_result,
            nli_result=self.mock_nli,
            generated_answer=self.answer,
            evidence_text=self.evidence,
            question=self.question,
            question_pmid="10001",
        )
        self.assertAlmostEqual(tf.features["n_unique_pmids"], 1.0, places=5)

    def test_five_candidates_duplicate_pmids(self):
        """Case 2: Five candidates with duplicate PMIDs (3 unique) should yield 3/5 = 0.6."""
        ret_result = {
            "top1_score": 0.035,
            "score_margin": 0.010,
            "candidate_pmids": ["10001", "10001", "10002", "10003", "10003"],
            "evidence_pmids": ["10001", "10002", "10003"],
        }
        tf = self.builder.build(
            retrieval_result=ret_result,
            nli_result=self.mock_nli,
            generated_answer=self.answer,
            evidence_text=self.evidence,
            question=self.question,
            question_pmid="10001",
        )
        self.assertAlmostEqual(tf.features["n_unique_pmids"], 0.6, places=5)

    def test_fewer_than_five_available_candidates(self):
        """Case 3: Fewer than five candidates (e.g., 2 candidates with 2 unique PMIDs) -> 2/5 = 0.4."""
        ret_result = {
            "top1_score": 0.035,
            "score_margin": 0.010,
            "candidate_pmids": ["10001", "10002"],
            "evidence_pmids": ["10001", "10002"],
        }
        tf = self.builder.build(
            retrieval_result=ret_result,
            nli_result=self.mock_nli,
            generated_answer=self.answer,
            evidence_text=self.evidence,
            question=self.question,
            question_pmid="10001",
        )
        self.assertAlmostEqual(tf.features["n_unique_pmids"], 0.4, places=5)

    def test_empty_candidate_list(self):
        """Case 4: Empty candidate list should return 0.0 without throwing an exception."""
        ret_result = {
            "top1_score": 0.0,
            "score_margin": 0.0,
            "candidate_pmids": [],
            "evidence_pmids": [],
        }
        tf = self.builder.build(
            retrieval_result=ret_result,
            nli_result=self.mock_nli,
            generated_answer=self.answer,
            evidence_text=self.evidence,
            question=self.question,
            question_pmid="10001",
        )
        self.assertAlmostEqual(tf.features["n_unique_pmids"], 0.0, places=5)

    def test_duplicate_chunks_from_one_pmid(self):
        """Case 5: All 5 candidates are chunks from a single PMID -> 1/5 = 0.2."""
        ret_result = {
            "top1_score": 0.035,
            "score_margin": 0.010,
            "candidate_pmids": ["10001", "10001", "10001", "10001", "10001"],
            "evidence_pmids": ["10001"],
        }
        tf = self.builder.build(
            retrieval_result=ret_result,
            nli_result=self.mock_nli,
            generated_answer=self.answer,
            evidence_text=self.evidence,
            question=self.question,
            question_pmid="10001",
        )
        self.assertAlmostEqual(tf.features["n_unique_pmids"], 0.2, places=5)

    def test_separation_of_candidate_pool_and_generator_evidence(self):
        """Case 6: Candidate pool has 4 unique PMIDs, while generator evidence has 3 passages.
        The feature must be computed strictly from the candidate pool (4/5 = 0.8), not the generator evidence (0.6).
        """
        ret_result = {
            "top1_score": 0.035,
            "score_margin": 0.010,
            "candidate_pmids": ["10001", "10002", "10003", "10004", "10001"],  # 4 unique PMIDs
            "evidence_pmids": ["10001", "10002", "10003"],                     # 3 generator evidence passages
        }
        tf = self.builder.build(
            retrieval_result=ret_result,
            nli_result=self.mock_nli,
            generated_answer=self.answer,
            evidence_text=self.evidence,
            question=self.question,
            question_pmid="10001",
        )
        self.assertAlmostEqual(tf.features["n_unique_pmids"], 0.8, places=5)
        self.assertNotEqual(tf.features["n_unique_pmids"], 0.6)


if __name__ == "__main__":
    unittest.main()
