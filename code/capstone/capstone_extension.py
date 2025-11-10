from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load


@functools.lru_cache(None)
def load_capstone_module():
    """Compile (if needed) and return the CUDA extension."""
    root = Path(__file__).resolve().parent
    src = root / "capstone_kernels.cu"

    extra_cuda_cflags = [
        "-std=c++20",
        "-gencode=arch=compute_121,code=sm_121",
        "-lineinfo",
    ]
    extra_cflags = ["-std=c++20"]

    module = load(
        name="blackwell_capstone_ext",
        sources=[str(src)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module
