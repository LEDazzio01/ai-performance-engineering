"""Torch extension loader for TMEM-backed KV cache epilogues."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load

_BUILD_ERROR: Optional[Exception] = None


@functools.lru_cache(None)
def load_tmem_cache_module():
    """Compile and load the TMEM cache extension once per process."""
    global _BUILD_ERROR
    src = Path(__file__).with_name("tmem_cache_ext.cu")
    try:
        repo_root = src.resolve().parents[2]
        include_dirs = [
            repo_root / "common" / "headers",
            repo_root / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
            repo_root / "third_party" / "cutlass" / "include",
        ]
        return load(
            name="kv_cache_tmem_ext",
            sources=[str(src)],
            extra_cuda_cflags=[
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ]
            + [f"-I{p}" for p in include_dirs],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - build failures are surfaced to callers
        _BUILD_ERROR = exc
        raise


def build_error() -> Optional[Exception]:
    """Return the cached build failure, if any."""
    return _BUILD_ERROR


__all__ = ["load_tmem_cache_module", "build_error"]
