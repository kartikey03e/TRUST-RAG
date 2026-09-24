"""
TRUST-RAG V2 — Configuration Loader
====================================
Loads YAML configs, sets random seeds, resolves paths relative to project root.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Project root (two levels up from src/config.py) ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Default config path ─────────────────────────────────────────────────────
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML configuration with optional overrides.

    Parameters
    ----------
    config_path : str | Path | None
        Path to a YAML config file.  Falls back to ``config/default.yaml``.
    overrides : dict | None
        Key-value pairs that override loaded config values.  Supports nested
        keys with dot notation, e.g. ``{"retrieval.final_top_k": 3}``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not available — fine for CPU-only stages


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── Convenience: load default config on import ──────────────────────────────
def get_default_config() -> dict[str, Any]:
    """Load and return the default configuration."""
    return load_config()
