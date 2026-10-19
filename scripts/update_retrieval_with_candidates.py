"""
Update retrieval JSONL cache files with candidate pool PMIDs.
Retains exact top-3 generator evidence passages while adding candidate_pmids (retained top-5 fused candidates).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config, resolve_path
from src.retriever import HybridRetriever
from src.utils import get_logger, load_jsonl, save_jsonl, timer

logger = get_logger("update_retrieval")


def update_retrieval_cache():
    cfg = get_default_config()
    results_dir = resolve_path(cfg["results"]["base_dir"])
    ret_dir = results_dir / "retrieval"
    ret_dir.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever.from_config(cfg)

    for split in ["val", "train"]:
        data_path = resolve_path(cfg["data"]["processed_dir"]) / f"{split}.jsonl"
        ret_cache_path = ret_dir / f"{split}_retrieval.jsonl"
        records = load_jsonl(data_path)
        logger.info(f"Processing {len(records)} records for {split} split...")

        existing_entries = {}
        if ret_cache_path.exists():
            for item in load_jsonl(ret_cache_path):
                existing_entries[str(item["question_pmid"])] = item

        updated_entries = []
        with timer(f"Updating {split} retrieval cache", logger):
            for i, rec in enumerate(records, 1):
                pmid = str(rec["pmid"])
                res = retriever.retrieve_with_candidates(
                    query=rec["question"],
                    candidate_top_k=5,
                    evidence_top_k=3,
                    mode="hybrid",
                )

                # If existing entry has evidence texts/scores, preserve them or use res
                existing = existing_entries.get(pmid, {})
                entry = {
                    "question_pmid": pmid,
                    "candidate_pmids": res["candidate_pmids"],
                    "evidence_pmids": existing.get("evidence_pmids", res["evidence_pmids"]),
                    "evidence_scores": existing.get("evidence_scores", res["evidence_scores"]),
                    "evidence_texts": existing.get("evidence_texts", res["evidence_texts"]),
                    "n_chunks": existing.get("n_chunks", [b.n_chunks for b in res["evidence_blocks"]]),
                }
                updated_entries.append(entry)

                if i % 100 == 0 or i == len(records):
                    logger.info(f"  Progress ({split}): {i}/{len(records)}")

        save_jsonl(updated_entries, ret_cache_path)
        logger.info(f"Saved {len(updated_entries)} updated retrieval entries to {ret_cache_path}")

    retriever.unload_query_encoder()
    logger.info("Retrieval update complete.")


if __name__ == "__main__":
    update_retrieval_cache()
