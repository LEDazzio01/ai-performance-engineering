"""Torch extension loader for TMEM-backed KV cache epilogues."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from core.utils.extension_loader_template import load_cuda_extension

_EXT_NAME = "kv_cache_tmem_ext"
_ROOT = Path(__file__).resolve().parent


@functools.lru_cache(None)
def load_tmem_cache_module():
    """Compile and load the TMEM cache extension once per process.
    
    Uses load_cuda_extension which automatically:
    - Adds TE CUTLASS and standalone CUTLASS include dirs
    
    IMPORTANT: We use the 'a' suffix in gencode (compute_100a, sm_100a) which
    defines __CUDA_ARCH_FEAT_SM100_ALL. This enables CUTE_ARCH_TCGEN05_TMEM_ENABLED
    in CUTLASS, which is required for TMEM operations. Without the 'a' suffix,
    TMEM code is compiled out!
    
    Note: We don't cache build errors because they may be transient.
    The lru_cache handles successful loads.
    """
    # SM100 (Blackwell) requires C++17 and SPECIFIC gencode flags with 'a' suffix.
    # The 'a' suffix (compute_100a) defines __CUDA_ARCH_FEAT_SM100_ALL which
    # enables CUTE_ARCH_TCGEN05_TMEM_ENABLED in CUTLASS for TMEM support.
    cuda_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        # CRITICAL: 'a' suffix enables __CUDA_ARCH_FEAT_SM100_ALL for TMEM
        "-gencode=arch=compute_100a,code=sm_100a",   # Blackwell SM100
        "-gencode=arch=compute_103a,code=sm_103a",   # Blackwell Ultra SM103
    ]
    
    return load_cuda_extension(
        extension_name=_EXT_NAME,
        cuda_source_file=str(_ROOT / "tmem_cache_ext.cu"),
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=["-lcuda"],
    )


def build_error() -> Optional[Exception]:
    """Return the build error if load failed, None otherwise.
    
    Note: This attempts to load the module to check for errors.
    """
    try:
        load_tmem_cache_module()
        return None
    except Exception as exc:
        return exc


__all__ = ["load_tmem_cache_module", "build_error"]
