from __future__ import annotations

import functools
from pathlib import Path

from common.python.extension_loader_template import load_cuda_extension_v2

_EXT_NAME = "grace_blackwell_capstone_ext"
_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parents[1]


@functools.lru_cache(None)
def load_grace_blackwell_module():
    """Compile (if needed) and return the CUDA 13 extension."""
    te_cutlass = _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include"
    upstream_cutlass = _REPO_ROOT / "third_party" / "cutlass" / "include"

    include_flags = []
    if te_cutlass.exists():
        include_flags.append(f"-I{te_cutlass}")
    include_flags.append(f"-I{upstream_cutlass}")

    return load_cuda_extension_v2(
        name=_EXT_NAME,
        sources=[_ROOT / "grace_blackwell_kernels.cu"],
        extra_cuda_cflags=[
            "-std=c++20",
            "-gencode=arch=compute_103,code=sm_103",
            "-gencode=arch=compute_100,code=sm_100",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_121,code=sm_121",
            "-gencode=arch=compute_103,code=compute_103",
            "-gencode=arch=compute_120,code=compute_120",
            "-gencode=arch=compute_121,code=compute_121",
            "-lineinfo",
            "-Xptxas=-v",
        ] + include_flags,
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
    )
