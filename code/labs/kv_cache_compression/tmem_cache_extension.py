"""Torch extension loader for TMEM-backed KV cache epilogues."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from common.python.extension_loader_template import load_cuda_extension_v2

_BUILD_ERROR: Optional[Exception] = None
_EXT_NAME = "kv_cache_tmem_ext"
_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parents[1]


@functools.lru_cache(None)
def load_tmem_cache_module():
    """Compile and load the TMEM cache extension once per process."""
    global _BUILD_ERROR
    
    include_dirs = [
        _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
        _REPO_ROOT / "common" / "headers",
        _REPO_ROOT / "third_party" / "cutlass" / "include",
    ]
    
    try:
        return load_cuda_extension_v2(
            name=_EXT_NAME,
            sources=[_ROOT / "tmem_cache_ext.cu"],
            extra_cuda_cflags=[
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ] + [f"-I{p}" for p in include_dirs if p.exists()],
        )
    except Exception as exc:  # pragma: no cover - build failures are surfaced to callers
        _BUILD_ERROR = exc
        raise


def build_error() -> Optional[Exception]:
    """Return the cached build failure, if any."""
    return _BUILD_ERROR


__all__ = ["load_tmem_cache_module", "build_error"]
