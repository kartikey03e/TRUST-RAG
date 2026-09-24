"""
TRUST-RAG V2 — Hybrid Retriever
=================================
Retrieves evidence for a query using hybrid BM25 + MedCPT dense retrieval,
fuses rankings via RRF, groups by PMID, and returns top-K evidence blocks.

Usage
-----
    from src.retriever import HybridRetriever
    retriever = HybridRetriever.from_config(config)
    results = retriever.retrieve("Does X cause Y?", top_k=5)
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import get_default_config, resolve_path
from src.indexer import tokenize_for_bm25
from src.utils import clear_gpu_memory, get_logger, print_gpu_memory

logger = get_logger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedPassage:
    """A single retrieved passage with its metadata."""

    text: str
    pmid: str
    score: float
    rank: int
    source: str  # "bm25", "dense", or "fused"
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class EvidenceBlock:
    """A PMID-level evidence block aggregating multiple passages."""

    pmid: str
    text: str  # concatenated passage texts
    score: float  # aggregated score
    n_chunks: int
    passages: list[RetrievedPassage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "text": self.text,
            "score": round(self.score, 6),
            "n_chunks": self.n_chunks,
            "metadata": self.metadata,
        }


# ── Retriever ────────────────────────────────────────────────────────────────

class HybridRetriever:
    """Hybrid BM25 + MedCPT retriever with RRF fusion and PMID grouping."""

    def __init__(
        self,
        bm25_data: dict,
        faiss_index: faiss.IndexFlatIP,
        dense_model_name: str = "ncbi/MedCPT-Query-Encoder",
        rrf_k: int = 60,
        initial_top_k: int = 20,
        final_top_k: int = 5,
        pmid_score_agg: str = "max",
        device: str = "cpu",
    ):
        # BM25 components
        self.bm25 = bm25_data["bm25"]
        self.passages = bm25_data["passages"]
        self.pmids = bm25_data["pmids"]
        self.metadata = bm25_data["metadata"]

        # Dense components
        self.faiss_index = faiss_index
        self.dense_model_name = dense_model_name
        self._query_encoder = None
        self._query_tokenizer = None

        # Config
        self.rrf_k = rrf_k
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        self.pmid_score_agg = pmid_score_agg
        self.device = device

    @classmethod
    def from_config(cls, config: dict | None = None) -> "HybridRetriever":
        """Construct a HybridRetriever from a YAML config."""
        if config is None:
            config = get_default_config()

        ret_cfg = config["retrieval"]

        # Load BM25
        bm25_path = resolve_path(ret_cfg["bm25_index_path"])
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        logger.info(f"Loaded BM25 index: {len(bm25_data['passages'])} passages")

        # Load FAISS
        faiss_path = resolve_path(ret_cfg["faiss_index_path"])
        faiss_index = faiss.read_index(str(faiss_path))
        logger.info(f"Loaded FAISS index: {faiss_index.ntotal} vectors")

        from src.config import get_device

        return cls(
            bm25_data=bm25_data,
            faiss_index=faiss_index,
            dense_model_name=ret_cfg["dense_model"],
            rrf_k=ret_cfg["rrf_k"],
            initial_top_k=ret_cfg["initial_top_k"],
            final_top_k=ret_cfg["final_top_k"],
            pmid_score_agg=ret_cfg["pmid_score_aggregation"],
            device=get_device(),
        )

    # ── Dense encoding ───────────────────────────────────────────────────

    def _load_query_encoder(self) -> None:
        """Lazy-load the MedCPT query encoder."""
        if self._query_encoder is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading query encoder: {self.dense_model_name}")
        self._query_tokenizer = AutoTokenizer.from_pretrained(self.dense_model_name)
        self._query_encoder = (
            AutoModel.from_pretrained(self.dense_model_name).to(self.device).eval()
        )
        print_gpu_memory("after query encoder load")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query using MedCPT query encoder."""
        import torch

        self._load_query_encoder()

        encoded = self._query_tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._query_encoder(**encoded)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = torch.nn.functional.normalize(embedding, dim=1)

        return embedding.cpu().numpy().astype(np.float32)

    def unload_query_encoder(self) -> None:
        """Free query encoder from GPU memory."""
        del self._query_encoder, self._query_tokenizer
        self._query_encoder = None
        self._query_tokenizer = None
        clear_gpu_memory()
        logger.info("Query encoder unloaded")

    # ── BM25 retrieval ───────────────────────────────────────────────────

    def _retrieve_bm25(self, query: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve top-K passages via BM25."""
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(
                RetrievedPassage(
                    text=self.passages[idx],
                    pmid=self.pmids[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    source="bm25",
                    chunk_index=self.metadata[idx].get("chunk_index", 0),
                    metadata=self.metadata[idx],
                )
            )
        return results

    # ── Dense retrieval ──────────────────────────────────────────────────

    def _retrieve_dense(self, query: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve top-K passages via MedCPT dense retrieval."""
        query_embedding = self._encode_query(query)
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            results.append(
                RetrievedPassage(
                    text=self.passages[idx],
                    pmid=self.pmids[idx],
                    score=float(score),
                    rank=rank + 1,
                    source="dense",
                    chunk_index=self.metadata[idx].get("chunk_index", 0),
                    metadata=self.metadata[idx],
                )
            )
        return results

    # ── RRF Fusion ───────────────────────────────────────────────────────

    def _fuse_rrf(
        self,
        bm25_results: list[RetrievedPassage],
        dense_results: list[RetrievedPassage],
    ) -> list[RetrievedPassage]:
        """Fuse BM25 and dense rankings using Reciprocal Rank Fusion.

        RRF_score(d) = Σ 1 / (k + rank_s(d))
        """
        # Build passage index → RRF score mapping
        # Use (pmid, chunk_index) as key to deduplicate across retrievers
        rrf_scores: dict[tuple, float] = defaultdict(float)
        passage_map: dict[tuple, RetrievedPassage] = {}

        for passage in bm25_results:
            key = (passage.pmid, passage.chunk_index)
            rrf_scores[key] += 1.0 / (self.rrf_k + passage.rank)
            passage_map[key] = passage

        for passage in dense_results:
            key = (passage.pmid, passage.chunk_index)
            rrf_scores[key] += 1.0 / (self.rrf_k + passage.rank)
            if key not in passage_map:
                passage_map[key] = passage

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        fused = []
        for rank, key in enumerate(sorted_keys):
            p = passage_map[key]
            fused.append(
                RetrievedPassage(
                    text=p.text,
                    pmid=p.pmid,
                    score=rrf_scores[key],
                    rank=rank + 1,
                    source="fused",
                    chunk_index=p.chunk_index,
                    metadata=p.metadata,
                )
            )
        return fused

    # ── PMID grouping ────────────────────────────────────────────────────

    def _group_by_pmid(
        self, passages: list[RetrievedPassage], top_k: int
    ) -> list[EvidenceBlock]:
        """Group retrieved passages by PMID and aggregate scores."""
        groups: dict[str, list[RetrievedPassage]] = defaultdict(list)
        for p in passages:
            groups[p.pmid].append(p)

        blocks = []
        for pmid, group_passages in groups.items():
            # Aggregate score
            scores = [p.score for p in group_passages]
            if self.pmid_score_agg == "max":
                agg_score = max(scores)
            else:
                agg_score = sum(scores) / len(scores)

            # Concatenate texts (ordered by chunk_index)
            sorted_passages = sorted(group_passages, key=lambda p: p.chunk_index)
            combined_text = " ".join(p.text for p in sorted_passages)

            blocks.append(
                EvidenceBlock(
                    pmid=pmid,
                    text=combined_text,
                    score=agg_score,
                    n_chunks=len(group_passages),
                    passages=sorted_passages,
                    metadata=group_passages[0].metadata,
                )
            )

        # Sort by aggregated score and return top-K
        blocks.sort(key=lambda b: b.score, reverse=True)
        return blocks[:top_k]

    # ── Main retrieve method ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        return_passages: bool = False,
        mode: str = "hybrid",
    ) -> list[EvidenceBlock]:
        """Retrieve evidence for a query using hybrid, dense-only, or BM25-only retrieval.

        Parameters
        ----------
        query : str
            The biomedical question.
        top_k : int | None
            Number of PMID-level evidence blocks to return.
            Defaults to ``self.final_top_k``.
        return_passages : bool
            If True, include raw passages in EvidenceBlock.passages.
        mode : str
            Retrieval mode: "hybrid" (RRF fusion), "dense" (MedCPT only),
            or "bm25" (BM25 only).

        Returns
        -------
        list[EvidenceBlock]
            Top-K evidence blocks, grouped by PMID.
        """
        if top_k is None:
            top_k = self.final_top_k

        if mode == "bm25":
            passages = self._retrieve_bm25(query, self.initial_top_k)
            blocks = self._group_by_pmid(passages, top_k)
        elif mode == "dense":
            passages = self._retrieve_dense(query, self.initial_top_k)
            blocks = self._group_by_pmid(passages, top_k)
        else:
            # Stage 1: Retrieve from both sources
            bm25_results = self._retrieve_bm25(query, self.initial_top_k)
            dense_results = self._retrieve_dense(query, self.initial_top_k)

            # Stage 2: Fuse via RRF
            fused = self._fuse_rrf(bm25_results, dense_results)

            # Stage 3: Group by PMID
            blocks = self._group_by_pmid(fused, top_k)

        if not return_passages:
            for b in blocks:
                b.passages = []  # Don't carry raw passages to save memory

        return blocks

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> list[list[EvidenceBlock]]:
        """Retrieve evidence for a batch of queries.

        Returns
        -------
        list[list[EvidenceBlock]]
            One list of evidence blocks per query.
        """
        return [self.retrieve(q, top_k=top_k, mode=mode) for q in queries]
