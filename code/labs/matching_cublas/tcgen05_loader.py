"""
Self-contained tcgen05 kernel loader for the Matching cuBLAS lab.

This module JIT-compiles the tcgen05 GEMM kernel without depending on
any other chapter or common code.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_LAB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LAB_DIR.parents[1]  # labs/matching_cublas -> labs -> code

# CUTLASS include paths - check multiple possible locations
_CUTLASS_CANDIDATES = [
    _REPO_ROOT / "third_party" / "cutlass" / "include",  # Standalone CUTLASS with SM100 support
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "pytorch-src" / "third_party" / "fbgemm" / "external" / "cutlass" / "include",
]


def _find_cutlass_include() -> Path | None:
    """Find CUTLASS include directory."""
    for cand in _CUTLASS_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _get_cuda_flags() -> list[str]:
    """Get CUDA compiler flags for tcgen05."""
    flags = ["-std=c++20"]
    
    # Add CUTLASS include
    cutlass_inc = _find_cutlass_include()
    if cutlass_inc:
        flags.append(f"-I{cutlass_inc}")
    else:
        raise RuntimeError(
            "CUTLASS include directory not found. "
            "Please ensure third_party/cutlass is available."
        )
    
    # SM100a for Blackwell (enables TMEM/tcgen05)
    major, minor = torch.cuda.get_device_capability()
    if major >= 10:
        flags.append("-gencode=arch=compute_100a,code=sm_100a")
    else:
        raise RuntimeError(
            f"tcgen05 requires SM 10.0+ (Blackwell). "
            f"Current GPU is SM {major}.{minor}"
        )
    
    return flags


def _load_kernel(source_file: Path, name_prefix: str):
    """Generic kernel loader."""
    if not source_file.exists():
        raise FileNotFoundError(f"{source_file.name} not found in {_LAB_DIR}")
    
    cuda_flags = _get_cuda_flags()
    src_hash = hashlib.md5(source_file.read_bytes()).hexdigest()[:8]
    build_name = f"{name_prefix}_{src_hash}"
    
    print(f"  [Compiling {source_file.name} (first time only)...]")
    module = load(
        name=build_name,
        sources=[str(source_file)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module


@lru_cache(maxsize=1)
def load_tcgen05_module():
    """JIT-compile and load the basic tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_gemm.cu", "lab_tcgen05")


@lru_cache(maxsize=1)
def load_tcgen05_pipelined_module():
    """JIT-compile and load the 2-stage pipelined tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_pipelined.cu", "lab_tcgen05_pipelined")


@lru_cache(maxsize=1)
def load_tcgen05_3stage_module():
    """JIT-compile and load the 3-stage pipelined tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_3stage.cu", "lab_tcgen05_3stage")


@lru_cache(maxsize=1)
def load_tcgen05_swizzled_module():
    """JIT-compile and load the swizzled tile scheduling tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_swizzled.cu", "lab_tcgen05_swizzled")


@lru_cache(maxsize=1)
def load_tcgen05_cluster_module():
    """JIT-compile and load the thread block cluster tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_cluster.cu", "lab_tcgen05_cluster")


@lru_cache(maxsize=1)
def load_tcgen05_persistent_module():
    """JIT-compile and load the persistent kernel tcgen05 GEMM."""
    return _load_kernel(_LAB_DIR / "tcgen05_persistent.cu", "lab_tcgen05_persistent")





def matmul_tcgen05(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM: C = A @ B^T
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
    
    Returns:
        MxN FP16 tensor
    """
    module = load_tcgen05_module()
    return module.matmul_tcgen05(a, b)


def matmul_tcgen05_bias_silu(
    a: torch.Tensor, 
    b: torch.Tensor, 
    bias: torch.Tensor
) -> torch.Tensor:
    """Execute tcgen05 GEMM with fused bias+SiLU epilogue.
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
        bias: N-element FP16 or FP32 bias vector
    
    Returns:
        MxN FP16 tensor with bias+SiLU applied
    """
    module = load_tcgen05_module()
    return module.matmul_tcgen05_bias_silu(a, b, bias)


def matmul_tcgen05_pipelined(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 2-stage pipelined tcgen05 GEMM: C = A @ B^T
    
    Overlaps TMA loads of tile K+1 with compute of tile K.
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
    
    Returns:
        MxN FP16 tensor
    """
    module = load_tcgen05_pipelined_module()
    return module.matmul_tcgen05_pipelined(a, b)


def matmul_tcgen05_3stage(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 3-stage pipelined tcgen05 GEMM: C = A @ B^T
    
    Deeper pipelining with 3 shared memory buffers.
    Prefetches 2 tiles ahead while computing current tile.
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
    
    Returns:
        MxN FP16 tensor
    """
    module = load_tcgen05_3stage_module()
    return module.matmul_tcgen05_3stage(a, b)


def matmul_tcgen05_swizzled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute swizzled tcgen05 GEMM: C = A @ B^T
    
    3-stage pipeline with swizzled tile scheduling for L2 optimization.
    Tiles are processed in cache-friendly order using XOR swizzle.
    """
    module = load_tcgen05_swizzled_module()
    return module.matmul_tcgen05_swizzled(a, b)


def matmul_tcgen05_cluster(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with 2x1 thread block clusters: C = A @ B^T
    
    Uses thread block clusters for better L2 cache utilization.
    2 CTAs along M dimension share a cluster, improving A matrix reuse.
    """
    module = load_tcgen05_cluster_module()
    return module.matmul_tcgen05_cluster(a, b)


def matmul_tcgen05_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute persistent kernel tcgen05 GEMM: C = A @ B^T
    
    CTAs stay resident and process multiple output tiles:
    - Launch one CTA per SM
    - Work-stealing for load balancing
    - Better L2 locality between consecutive tiles
    """
    module = load_tcgen05_persistent_module()
    return module.matmul_tcgen05_persistent(a, b)

