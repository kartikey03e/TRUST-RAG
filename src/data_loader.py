"""
TRUST-RAG V2 — Data Loader
============================
Loads PubMedQA CSV, computes dataset statistics, creates stratified
train/val/test splits, and serialises processed splits to JSONL.

Usage
-----
    python -m src.data_loader                     # uses default config
    python -m src.data_loader --config config.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_default_config, load_config, resolve_path, set_seed
from src.utils import get_logger, save_json, save_jsonl

logger = get_logger(__name__)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_pubmedqa(csv_path: str | Path) -> pd.DataFrame:
    """Load the PubMedQA expert-labeled CSV and parse list-valued columns.

    Parameters
    ----------
    csv_path : str | Path
        Path to ``pubmedqa_pqal.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed ``contexts``, ``labels``, ``meshes`` columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns from {path.name}")

    # Parse stringified lists safely
    for col in ("contexts", "labels", "meshes"):
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)

    # Normalise decision labels to lowercase
    if "final_decision" in df.columns:
        df["final_decision"] = df["final_decision"].str.strip().str.lower()

    return df


def _safe_parse_list(val: Any) -> list | Any:
    """Parse a string that looks like a Python list; return as-is otherwise."""
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return val


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute and log key dataset statistics.

    Returns
    -------
    dict
        Dictionary with shape, class distribution, missing values, etc.
    """
    stats: dict[str, Any] = {}

    # Shape
    stats["n_rows"], stats["n_cols"] = df.shape
    stats["columns"] = list(df.columns)

    # Class distribution
    if "final_decision" in df.columns:
        dist = df["final_decision"].value_counts().to_dict()
        pct = df["final_decision"].value_counts(normalize=True).mul(100).round(2).to_dict()
        stats["class_distribution"] = dist
        stats["class_pct"] = pct
        logger.info(f"Class distribution: {dist}")
        logger.info(f"Class percentages:  {pct}")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    stats["missing_values"] = missing
    if missing:
        logger.info(f"Missing values: {missing}")

    # Duplicates — only check on hashable columns (exclude parsed lists)
    hashable_cols = [
        col for col in df.columns
        if not df[col].apply(lambda x: isinstance(x, (list, dict, set))).any()
    ]
    n_dup = df[hashable_cols].duplicated().sum() if hashable_cols else 0
    stats["n_duplicates"] = int(n_dup)
    logger.info(f"Duplicate rows: {n_dup}")

    # Year range
    if "year" in df.columns:
        valid_years = df["year"].dropna()
        if len(valid_years) > 0:
            stats["year_range"] = [int(valid_years.min()), int(valid_years.max())]
            stats["n_missing_years"] = int(df["year"].isnull().sum())

    # Context lengths
    if "contexts" in df.columns:
        ctx_lens = df["contexts"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        stats["context_count"] = {
            "mean": round(float(ctx_lens.mean()), 2),
            "min": int(ctx_lens.min()),
            "max": int(ctx_lens.max()),
        }

    return stats


# ── Splitting ────────────────────────────────────────────────────────────────

def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_col: str = "final_decision",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create stratified train/val/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    train_ratio, val_ratio, test_ratio : float
        Split proportions (must sum to 1.0).
    seed : int
        Random seed for reproducibility.
    stratify_col : str
        Column to stratify on.

    Returns
    -------
    tuple
        (train_df, val_df, test_df, split_info)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    # First split: train vs. (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=df[stratify_col],
    )

    # Second split: val vs. test
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=val_test_df[stratify_col],
    )

    # Record split indices for reproducibility
    split_info = {
        "seed": seed,
        "stratify_col": stratify_col,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "train_indices": sorted(train_df.index.tolist()),
        "val_indices": sorted(val_df.index.tolist()),
        "test_indices": sorted(test_df.index.tolist()),
    }

    # Log distribution per split
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df[stratify_col].value_counts().to_dict()
        logger.info(f"{name.upper()} ({len(split_df)} rows): {dist}")

    return train_df, val_df, test_df, split_info


# ── Serialization ────────────────────────────────────────────────────────────

def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, handling list columns."""
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val) if not np.isnan(val) else None
            elif isinstance(val, np.ndarray):
                val = val.tolist()
            record[col] = val
        records.append(record)
    return records


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_info: dict,
    output_dir: str | Path,
) -> None:
    """Save processed splits as JSONL and split metadata as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = output_dir / f"{name}.jsonl"
        save_jsonl(df_to_records(split_df), path)
        logger.info(f"Saved {name} split: {len(split_df)} records → {path}")

    splits_path = output_dir / "splits.json"
    # Don't save full indices to JSON if too large — save just metadata
    meta = {k: v for k, v in split_info.items() if k not in ("train_indices", "val_indices", "test_indices")}
    save_json(meta, splits_path)

    # Save full indices separately
    indices_path = output_dir / "split_indices.json"
    save_json(
        {
            "train_indices": split_info["train_indices"],
            "val_indices": split_info["val_indices"],
            "test_indices": split_info["test_indices"],
        },
        indices_path,
    )
    logger.info(f"Saved split metadata → {splits_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(config: dict | None = None) -> dict[str, Any]:
    """Execute the full data loading and splitting pipeline.

    Returns
    -------
    dict
        Contains 'stats', 'split_info', and paths to saved files.
    """
    if config is None:
        config = get_default_config()

    seed = config.get("seed", 42)
    set_seed(seed)

    data_cfg = config["data"]
    csv_path = resolve_path(data_cfg["raw_path"])
    output_dir = resolve_path(data_cfg["processed_dir"])

    # Load
    df = load_pubmedqa(csv_path)

    # Stats
    stats = compute_dataset_stats(df)

    # Split
    ratios = data_cfg["split_ratios"]
    train_df, val_df, test_df, split_info = create_stratified_splits(
        df,
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
        seed=seed,
    )

    # Save
    save_splits(train_df, val_df, test_df, split_info, output_dir)

    # Save stats
    stats_path = output_dir / "dataset_stats.json"
    save_json(stats, stats_path)
    logger.info(f"Saved dataset stats → {stats_path}")

    return {
        "stats": stats,
        "split_info": split_info,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRUST-RAG Data Loader")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    result = run(cfg)

    print("\n" + "=" * 60)
    print("DATASET STATS")
    print("=" * 60)
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("SPLIT SIZES")
    print("=" * 60)
    for k, v in result["split_info"]["sizes"].items():
        print(f"  {k}: {v}")
