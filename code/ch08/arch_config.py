"""Chapter 8 helpers for configuring PyTorch on Blackwell-era GPUs.

This module keeps the PyTorch demos aligned with the book narrative by:
  * enabling TF32 Tensor Core math by default (as recommended for training/profiling)
  * exposing a small wrapper around torch.compile that can be toggled via env vars

Environment variables:
  USE_COMPILE / TORCH_USE_COMPILE    -> 0/1 toggle for torch.compile (default: 0)
  COMPILE_MODE / TORCH_COMPILE_MODE  -> torch.compile mode name (default: reduce-overhead)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

from core.utils.compile_utils import compile_callable
from core.utils.compile_utils import enable_tf32


enable_tf32()

# Configure torch inductor cache directory to avoid C++ compilation errors
def _configure_inductor_cache() -> None:
    """Set up torch inductor cache directory to avoid C++ compilation errors."""
    cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        # Use absolute path in repo root to avoid working directory issues
        repo_root = Path(__file__).parent.parent
        cache_path = repo_root / ".torch_inductor"
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
    else:
        # User-specified cache directory - ensure it exists
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            # Relative path - create in current working directory
            cache_path = Path.cwd() / cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
    
    # Create subdirectories needed by PyTorch inductor for C++ compilation
    # 'od' is for output directory, 'tk' is for temporary kernel files
    cache_path = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    try:
        (cache_path / "od").mkdir(parents=True, exist_ok=True)
        (cache_path / "tk").mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass  # If we can't create subdirectories, PyTorch will handle it

_configure_inductor_cache()

_VALID_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse an integer-like environment flag safely."""
    raw = os.getenv(name, default)
    try:
        return bool(int(raw))
    except (TypeError, ValueError):
        warnings.warn(f"Expected {name} to be 0 or 1; got {raw!r}. Treating as disabled.")
        return False


def should_use_compile() -> bool:
    """Return True if torch.compile should be applied based on env toggles."""
    if not hasattr(torch, "compile"):
        return False
    return _env_flag("USE_COMPILE", os.getenv("TORCH_USE_COMPILE", "0"))


def get_compile_mode(default: str = "reduce-overhead") -> str:
    """Return the requested torch.compile mode, falling back to a safe default."""
    mode = os.getenv("COMPILE_MODE", os.getenv("TORCH_COMPILE_MODE", default))
    if mode not in _VALID_MODES:
        warnings.warn(
            f"Unsupported torch.compile mode {mode!r}; "
            f"falling back to {default!r}."
        )
        return default
    return mode


def maybe_compile(fn, *, default_mode: str = "reduce-overhead"):
    """Return a compiled version of `fn` if torch.compile is requested and available."""
    if not should_use_compile():
        return fn

    mode = get_compile_mode(default_mode)
    return compile_callable(fn, mode=mode)
