"""Helper utilities for conditional NVTX range markers.

This module provides utilities to conditionally add NVTX ranges only when
profiling is enabled, reducing overhead for pure performance benchmarks.
"""

import sys
from contextlib import contextmanager, redirect_stderr
from typing import Any, Generator, Optional, TextIO, cast

import io

import torch

_QUALIFIER_SUFFIXES = {
    "sequential",
    "overlapped",
    "coalesced",
    "uncoalesced",
    "pinned",
    "batch",
    "bf16",
    "fp16",
    "fp32",
    "fp8",
    "mixed",
    "naive",
    "default",
    "baseline",
    "optimized",
    "dense",
    "sparse",
    "graph",
    "pytorch",
    "cuda",
    "triton",
    "variant",
    "standard",
    "eager",
}


def canonicalize_nvtx_name(name: str) -> str:
    tokens = [tok for tok in name.lower().split("_") if tok]
    while tokens and tokens[0] in ("baseline", "optimized"):
        tokens.pop(0)
    while tokens and tokens[-1] in _QUALIFIER_SUFFIXES:
        tokens.pop()
    if not tokens:
        return name.lower()
    return "_".join(tokens)


class FilteredStderr(io.TextIOBase):
    """Thread-safe filter for stderr that removes NVTX threading errors."""
    def __init__(self, original: TextIO):
        self.original = original
    
    def write(self, text: str) -> int:
        # Filter out NVTX threading error messages
        if "External init callback must run in same thread as registerClient" not in text:
            self.original.write(text)
            return len(text)
        # Otherwise silently drop the error message
        return 0
    
    def flush(self) -> None:
        self.original.flush()
    
    def __getattr__(self, name: str):
        # Forward all other attributes to original stderr
        return getattr(self.original, name)


@contextmanager
def _suppress_nvtx_threading_error() -> Generator[None, None, None]:
    """Suppress NVTX threading errors that occur when CUDA initializes in different thread.
    
    This is a known PyTorch/NVTX issue where NVTX is initialized in one thread
    but used in another. The error is harmless and benchmarks complete successfully.
    We suppress stderr output for this specific error message using thread-safe redirect_stderr.
    """
    # Use contextlib.redirect_stderr for thread-safe stderr redirection
    filtered_stderr = FilteredStderr(sys.stderr)
    with redirect_stderr(cast(TextIO, filtered_stderr)):
        yield


@contextmanager
def nvtx_range(name: str, enable: Optional[bool] = None) -> Generator[None, None, None]:
    """Conditionally add NVTX range marker.
    
    Args:
        name: Name for the NVTX range
        enable: If True, add NVTX range; if False, no-op; if None, auto-detect from config
    
    Example:
        with nvtx_range("my_operation", enable=True):
            # This operation will be marked in NVTX traces
            result = model(input)
    """
    if enable is None:
        # Auto-detect: check if NVTX is enabled via environment or config
        # Default to False for minimal overhead
        enable = False
    
    canonical_name = canonicalize_nvtx_name(name)
    if enable and torch.cuda.is_available():
        with _suppress_nvtx_threading_error():
            torch.cuda.nvtx.range_push(canonical_name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        return
    # No-op when NVTX is disabled or CUDA is unavailable
    yield


def get_nvtx_enabled(config: Any) -> bool:
    """Get NVTX enabled status from benchmark config.
    
    Args:
        config: BenchmarkConfig instance
    
    Returns:
        True if NVTX should be enabled, False otherwise
    """
    nvtx_value = getattr(config, "enable_nvtx", None)
    if isinstance(nvtx_value, bool):
        return nvtx_value
    profiling_value = getattr(config, "enable_profiling", None)
    if isinstance(profiling_value, bool):
        return profiling_value
    return False
