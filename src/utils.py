"""
TRUST-RAG V2 — Utility Helpers
===============================
Logging, timing, I/O, and common helpers used across all modules.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import jsonlines


# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Timing ───────────────────────────────────────────────────────────────────

@contextmanager
def timer(description: str, logger: logging.Logger | None = None):
    """Context manager that logs elapsed time for a block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"{description}: {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ── JSONL I/O ────────────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Save a list of dicts as strict JSONL (one JSON object per line)."""
    import math

    def sanitize(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_records = [sanitize(record) for record in records]

    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(clean_records)
    


def load_jsonl(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file."""
    with jsonlines.open(Path(path), mode="r") as reader:
        return list(reader)


def append_jsonl(record: dict, path: str | Path) -> None:
    """Append a single record to a JSONL file (crash-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="a") as writer:
        writer.write(record)


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def save_json(obj: Any, path: str | Path) -> None:
    """Save a JSON-serializable object to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(Path(path), "r") as f:
        return json.load(f)


# ── GPU Memory ───────────────────────────────────────────────────────────────

def print_gpu_memory(label: str = "") -> None:
    """Print current GPU memory usage (if torch + CUDA available)."""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(
                f"[GPU {label}] "
                f"Allocated: {allocated:.2f} GB | "
                f"Reserved: {reserved:.2f} GB | "
                f"Total: {total:.2f} GB"
            )
    except ImportError:
        pass


def clear_gpu_memory() -> None:
    """Aggressively free GPU memory."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
