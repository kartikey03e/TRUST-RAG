"""
TRUST-RAG V2 — Evidence Indexer
================================
Builds a BM25 sparse index and a MedCPT dense FAISS index over the
PubMedQA context passages.  Each index entry is tagged with its PMID.

Usage
-----
    python -m src.indexer                  # uses default config
    python -m src.indexer --config my.yaml
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import get_default_config, load_config, resolve_path, set_seed
from src.utils import (
    clear_gpu_memory,
    get_logger,
    load_jsonl,
    print_gpu_memory,
    save_json,
    timer,
)

logger = get_logger(__name__)


# ── Text preprocessing ──────────────────────────────────────────────────────

def tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ── Document extraction ──────────────────────────────────────────────────────

def extract_documents(
    records: list[dict],
) -> tuple[list[str], list[str], list[dict]]:
    """Extract passages from JSONL records.

    Each PubMedQA record can have multiple context passages (list).
    We flatten them into individual passages while tracking the source PMID.

    Returns
    -------
    tuple
        (passages, pmids, metadata_list)
        - passages: list of text strings
        - pmids: list of PMID strings (one per passage)
        - metadata_list: list of dicts with passage-level metadata
    """
    passages: list[str] = []
    pmids: list[str] = []
    metadata: list[dict] = []

    for rec in records:
        pmid = str(rec.get("pmid", ""))
        contexts = rec.get("contexts", [])

        # If contexts is a list of strings, each is a passage
        if isinstance(contexts, list):
            for i, ctx in enumerate(contexts):
                if isinstance(ctx, str) and ctx.strip():
                    passages.append(ctx.strip())
                    pmids.append(pmid)
                    metadata.append({
                        "pmid": pmid,
                        "chunk_index": i,
                        "question": rec.get("question", ""),
                        "final_decision": rec.get("final_decision", ""),
                    })
        elif isinstance(contexts, str) and contexts.strip():
            # Single string context
            passages.append(contexts.strip())
            pmids.append(pmid)
            metadata.append({
                "pmid": pmid,
                "chunk_index": 0,
                "question": rec.get("question", ""),
                "final_decision": rec.get("final_decision", ""),
            })

    logger.info(
        f"Extracted {len(passages)} passages from {len(records)} records "
        f"({len(set(pmids))} unique PMIDs)"
    )
    return passages, pmids, metadata


# ── BM25 Index ───────────────────────────────────────────────────────────────

def build_bm25_index(passages: list[str]) -> BM25Okapi:
    """Build a BM25 index from tokenized passages."""
    with timer("BM25 index construction", logger):
        tokenized = [tokenize_for_bm25(p) for p in passages]
        bm25 = BM25Okapi(tokenized)
    logger.info(f"BM25 index built with {len(passages)} documents")
    return bm25


def save_bm25_index(
    bm25: BM25Okapi,
    passages: list[str],
    pmids: list[str],
    metadata: list[dict],
    path: str | Path,
) -> None:
    """Serialize BM25 index with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "bm25": bm25,
        "passages": passages,
        "pmids": pmids,
        "metadata": metadata,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"BM25 index saved → {path}")


def load_bm25_index(path: str | Path) -> dict:
    """Load BM25 index + metadata."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)


# ── Dense (MedCPT) Index ────────────────────────────────────────────────────

def build_dense_index(
    passages: list[str],
    model_name: str = "ncbi/MedCPT-Article-Encoder",
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Encode passages using MedCPT article encoder and return embeddings.

    Parameters
    ----------
    passages : list[str]
        Passage texts to encode.
    model_name : str
        HuggingFace model name for the article encoder.
    batch_size : int
        Encoding batch size.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    np.ndarray
        Normalized embeddings of shape (n_passages, dim).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading dense encoder: {model_name}")
    print_gpu_memory("before dense encoder")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    print_gpu_memory("after dense encoder load")

    all_embeddings = []
    with timer("Dense encoding", logger):
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)
                # MedCPT uses the [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]

            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

            if (start // batch_size) % 10 == 0:
                logger.info(
                    f"  Encoded {start + len(batch)}/{len(passages)} passages"
                )

    # Cleanup
    del model, tokenizer
    clear_gpu_memory()

    embeddings_array = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"Dense embeddings shape: {embeddings_array.shape}")
    return embeddings_array


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS flat inner-product index from normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: str | Path) -> None:
    """Save FAISS index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info(f"FAISS index saved → {path}")


def load_faiss_index(path: str | Path) -> faiss.IndexFlatIP:
    """Load FAISS index from disk."""
    return faiss.read_index(str(path))


# ── Main ─────────────────────────────────────────────────────────────────────

def run(config: dict | None = None) -> dict[str, Any]:
    """Build and save both BM25 and FAISS indices.

    Returns
    -------
    dict
        Paths to saved indices and document counts.
    """
    if config is None:
        config = get_default_config()

    set_seed(config.get("seed", 42))
    ret_cfg = config["retrieval"]
    data_cfg = config["data"]

    # Load ALL records (train + val + test) — index the full corpus
    processed_dir = resolve_path(data_cfg["processed_dir"])
    all_records = []
    for split in ("train", "val", "test"):
        split_path = processed_dir / f"{split}.jsonl"
        if split_path.exists():
            all_records.extend(load_jsonl(split_path))
    logger.info(f"Loaded {len(all_records)} total records for indexing")

    # Extract passages
    passages, pmids, metadata = extract_documents(all_records)

    # ── BM25 ──
    bm25 = build_bm25_index(passages)
    bm25_path = resolve_path(ret_cfg["bm25_index_path"])
    save_bm25_index(bm25, passages, pmids, metadata, bm25_path)

    # ── Dense (MedCPT) ──
    from src.config import get_device

    device = get_device()
    embeddings = build_dense_index(
        passages,
        model_name=ret_cfg["dense_article_model"],
        device=device,
    )
    faiss_idx = build_faiss_index(embeddings)
    faiss_path = resolve_path(ret_cfg["faiss_index_path"])
    save_faiss_index(faiss_idx, faiss_path)

    # ── Save metadata mapping ──
    pmid_meta_path = resolve_path(ret_cfg["pmid_metadata_path"])
    save_json(
        {"pmids": pmids, "n_passages": len(passages), "n_unique_pmids": len(set(pmids))},
        pmid_meta_path,
    )

    result = {
        "n_passages": len(passages),
        "n_unique_pmids": len(set(pmids)),
        "bm25_path": str(bm25_path),
        "faiss_path": str(faiss_path),
        "embedding_dim": embeddings.shape[1],
    }
    logger.info(f"Indexing complete: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRUST-RAG Evidence Indexer")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    run(cfg)
