"""
TRUST-RAG V2 — Retrieval Evaluation
=====================================
Computes Recall@K, MRR, and generates retrieval diagnostic reports.
Uses gold PMID to determine retrieval success (no answer leakage).

Usage
-----
    from src.retrieval_eval import evaluate_retrieval
    metrics = evaluate_retrieval(retriever, records, ks=[1, 3, 5])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.retriever import EvidenceBlock, HybridRetriever
from src.utils import get_logger, save_json, timer

logger = get_logger(__name__)


def evaluate_single_query(
    gold_pmid: str,
    retrieved_blocks: list[EvidenceBlock],
    ks: list[int] = [1, 3, 5],
) -> dict[str, Any]:
    """Evaluate retrieval for a single query.

    Parameters
    ----------
    gold_pmid : str
        The correct PMID for this question.
    retrieved_blocks : list[EvidenceBlock]
        Retrieved evidence blocks (already PMID-grouped).
    ks : list[int]
        Values of K for Recall@K.

    Returns
    -------
    dict
        Per-query retrieval metrics.
    """
    retrieved_pmids = [b.pmid for b in retrieved_blocks]
    gold_pmid = str(gold_pmid)

    # Find rank of gold PMID (1-indexed, 0 if not found)
    gold_rank = 0
    for i, pmid in enumerate(retrieved_pmids):
        if str(pmid) == gold_pmid:
            gold_rank = i + 1
            break

    result = {
        "gold_pmid": gold_pmid,
        "retrieved_pmids": retrieved_pmids,
        "gold_rank": gold_rank,
        "gold_found": gold_rank > 0,
    }

    # Recall@K
    for k in ks:
        result[f"recall_at_{k}"] = 1.0 if 0 < gold_rank <= k else 0.0

    # Reciprocal Rank
    result["reciprocal_rank"] = 1.0 / gold_rank if gold_rank > 0 else 0.0

    # Top-1 score
    result["top1_score"] = retrieved_blocks[0].score if retrieved_blocks else 0.0

    # Score margin (top-1 minus top-2)
    if len(retrieved_blocks) >= 2:
        result["score_margin"] = retrieved_blocks[0].score - retrieved_blocks[1].score
    else:
        result["score_margin"] = 0.0

    return result


def evaluate_retrieval(
    retriever: HybridRetriever,
    records: list[dict],
    ks: list[int] = [1, 3, 5],
    top_k: int = 5,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Evaluate retrieval over a set of records.

    Parameters
    ----------
    retriever : HybridRetriever
        The retriever to evaluate.
    records : list[dict]
        Dataset records with 'question' and 'pmid' fields.
    ks : list[int]
        Values of K for Recall@K.
    top_k : int
        Number of evidence blocks to retrieve per query.
    mode : str
        Retrieval mode: "hybrid", "dense", or "bm25".

    Returns
    -------
    dict
        Aggregate retrieval metrics and per-query results.
    """
    per_query_results = []

    with timer(f"Retrieval evaluation [{mode}] ({len(records)} queries)", logger):
        for i, rec in enumerate(records):
            question = rec["question"]
            gold_pmid = str(rec["pmid"])

            # Retrieve
            blocks = retriever.retrieve(question, top_k=top_k, mode=mode)

            # Evaluate
            qr = evaluate_single_query(gold_pmid, blocks, ks=ks)
            qr["question_index"] = i
            qr["question"] = question
            per_query_results.append(qr)

            if (i + 1) % 50 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(records)} queries")

    # Aggregate metrics
    aggregate = {}
    for k in ks:
        key = f"recall_at_{k}"
        values = [r[key] for r in per_query_results]
        aggregate[key] = round(float(np.mean(values)), 4)

    rr_values = [r["reciprocal_rank"] for r in per_query_results]
    aggregate["mrr"] = round(float(np.mean(rr_values)), 4)

    aggregate["n_queries"] = len(records)
    aggregate["n_found"] = sum(1 for r in per_query_results if r["gold_found"])
    aggregate["n_not_found"] = aggregate["n_queries"] - aggregate["n_found"]

    # Log results
    logger.info("=" * 50)
    logger.info("RETRIEVAL EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in aggregate.items():
        logger.info(f"  {k}: {v}")

    return {
        "aggregate": aggregate,
        "per_query": per_query_results,
    }
