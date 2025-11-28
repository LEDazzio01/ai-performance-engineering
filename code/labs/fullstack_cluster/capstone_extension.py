from __future__ import annotations

import functools
from pathlib import Path

from common.python.extension_loader_template import load_cuda_extension_v2

_EXT_NAME = "blackwell_capstone_ext"
_ROOT = Path(__file__).resolve().parent


@functools.lru_cache(None)
def load_capstone_module():
    """Compile (if needed) and return the CUDA extension."""
    return load_cuda_extension_v2(
        name=_EXT_NAME,
        sources=[_ROOT / "capstone_kernels.cu"],
        extra_cuda_cflags=[
            "-std=c++20",
            "-gencode=arch=compute_100,code=sm_100",
            "-gencode=arch=compute_103,code=sm_103",
            "-gencode=arch=compute_121,code=sm_121",
            "-lineinfo",
        ],
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
    )
