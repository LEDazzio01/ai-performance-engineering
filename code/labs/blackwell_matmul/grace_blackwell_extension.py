from __future__ import annotations

import functools
from pathlib import Path

from torch.utils.cpp_extension import load


@functools.lru_cache(None)
def load_grace_blackwell_module():
    """Compile (if needed) and return the CUDA 13 extension."""
    root = Path(__file__).resolve().parent
    src = root / "grace_blackwell_kernels.cu"
    cutlass_include = root.parents[1] / "third_party" / "cutlass" / "include"

    extra_cuda_cflags = [
        "-std=c++20",
        "-gencode=arch=compute_103,code=sm_103",
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_120,code=sm_120",
        "-gencode=arch=compute_121,code=sm_121",
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_103,code=compute_103",
        "-gencode=arch=compute_120,code=compute_120",
        "-gencode=arch=compute_121,code=compute_121",
        "-lineinfo",
        "-Xptxas=-v",
        f"-I{cutlass_include}",
    ]
    extra_cflags = ["-std=c++20"]

    module = load(
        name="grace_blackwell_capstone_ext",
        sources=[str(src)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module
