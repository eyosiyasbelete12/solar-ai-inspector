"""Shared helpers: project paths, YAML config loading, reproducibility, device selection.

All scripts are meant to be run from the repository root, e.g.
    python -m scripts.check_setup
"""
from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Any, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PathLike = Union[str, Path]


def resolve_path(path: PathLike) -> Path:
    """Resolve a path from a config file. Relative paths are relative to the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: PathLike) -> dict[str, Any]:
    """Load a YAML file that must contain a mapping at the top level."""
    p = resolve_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {p}")
    return data


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (whichever are installed).

    deterministic=True also asks cuDNN for deterministic kernels (slower, and
    not every GPU op is bit-for-bit reproducible even then).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> str:
    """Return a torch-style device string: 'cuda', 'cuda:N', 'mps' or 'cpu'.

    'auto' picks CUDA, then Apple MPS, then CPU. If a specific device is requested
    but unavailable (or PyTorch is not installed) this falls back to CPU with a warning.
    """
    preference = (preference or "auto").lower()
    try:
        import torch
    except ImportError:
        if preference not in ("auto", "cpu"):
            warnings.warn(f"PyTorch not installed; ignoring device='{preference}' and using CPU.")
        return "cpu"

    if preference == "cpu":
        return "cpu"
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            return preference
        warnings.warn(f"device='{preference}' requested but CUDA is unavailable; using CPU.")
        return "cpu"
    if preference == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        warnings.warn("device='mps' requested but MPS is unavailable; using CPU.")
        return "cpu"
    raise ValueError(f"Unknown device '{preference}'. Use auto, cpu, cuda, cuda:N or mps.")
