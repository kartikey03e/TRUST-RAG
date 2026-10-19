"""
Regenerate training and validation trust feature files with corrected candidate pool PMIDs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config import get_default_config, resolve_path
from src.trust_features import FEATURE_NAMES, TrustFeatureBuilder
from src.utils import get_logger, load_jsonl, save_jsonl

logger = get_logger("regenerate_trust")


def regenerate_trust_features():
    cfg = get_default_config()
    results_dir = resolve_path(cfg["results"]["base_dir"])
    trust_dir = results_dir / "trust"
    trust_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        logger.info(f"Regenerating trust features for {split} split...")
        ret_path = results_dir / "retrieval" / f"{split}_retrieval.jsonl"
        ret_data = {str(r["question_pmid"]): r for r in load_jsonl(ret_path)}

        trust_path = trust_dir / f"qwen_{split}_trust.jsonl"
        existing_trust = load_jsonl(trust_path)
        logger.info(f"Loaded {len(existing_trust)} existing records from {trust_path}")

        updated_records = []
        unique_pmid_feature_values = []

        for rec in existing_trust:
            pmid = str(rec["question_pmid"])
            ret = ret_data.get(pmid)
            if not ret or "candidate_pmids" not in ret:
                raise ValueError(f"Missing candidate_pmids for PMID {pmid} in {ret_path}")

            candidate_pmids = ret["candidate_pmids"]
            # Deduplicate chunks from the same PMID
            n_unique = len(set(candidate_pmids))
            n_pmids_norm = round(min(n_unique / 5.0, 1.0), 6)

            updated_rec = dict(rec)
            updated_rec["n_unique_pmids"] = n_pmids_norm
            updated_records.append(updated_rec)
            unique_pmid_feature_values.append(n_pmids_norm)

        # Save updated trust records
        save_jsonl(updated_records, trust_path)
        logger.info(f"Saved {len(updated_records)} updated records to {trust_path}")

        vals = np.array(unique_pmid_feature_values)
        unique_vals, counts = np.unique(vals, return_counts=True)
        logger.info(f"{split.upper()} n_unique_pmids statistics:")
        logger.info(f"  Count: {len(vals)}")
        logger.info(f"  Unique values: {len(unique_vals)} -> {dict(zip(np.round(unique_vals, 4), counts))}")
        logger.info(f"  Min: {vals.min():.4f}, Max: {vals.max():.4f}, Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")


if __name__ == "__main__":
    regenerate_trust_features()
