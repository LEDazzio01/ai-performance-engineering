"""TMA (Tensor Memory Accelerator) prefetch extension for KV cache.

TMA is Blackwell's async bulk data movement engine for GMEM ↔ SMEM transfers.
This extension demonstrates prefetching KV cache tiles to hide memory latency.

Key concepts:
- TMA handles large bulk transfers efficiently without CPU involvement
- cp.async enables non-blocking copies that overlap with computation
- Double-buffering hides latency by prefetching the next tile while computing

Note: TMA is for data MOVEMENT. TMEM is for Tensor Core ACCUMULATORS.
They are complementary but serve different purposes.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from core.utils.extension_loader_template import load_cuda_extension

_EXT_NAME = "kv_cache_tma_ext"
_ROOT = Path(__file__).resolve().parent


@functools.lru_cache(None)
def load_tma_prefetch_module():
    """Compile and load the TMA prefetch extension once per process.

    Uses cp.async intrinsics for SM90+ async memory copy.
    Falls back to synchronous copy on older architectures.
    """
    cuda_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        # Support SM80 (Ampere), SM90 (Hopper) and SM100 (Blackwell)
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_100,code=sm_100",
    ]

    return load_cuda_extension(
        extension_name=_EXT_NAME,
        cuda_source_file=str(_ROOT / "tma_prefetch_ext.cu"),
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=["-lcuda"],
    )


def build_error() -> Optional[Exception]:
    """Return the build error if load failed, None otherwise."""
    try:
        load_tma_prefetch_module()
        return None
    except Exception as exc:
        return exc


__all__ = ["load_tma_prefetch_module", "build_error"]

