"""Shared tcgen05 kernel loaders and Python wrappers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load

from common.python.tcgen05_requirements import ensure_tcgen05_supported

try:  # Ensure TORCH_CUDA_ARCH_LIST stays clamped for GB-series hosts.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional bootstrap
    arch_config = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CUTLASS_INCLUDE = _REPO_ROOT / "third_party" / "cutlass" / "include"
_LEGACY_CUTLASS_INCLUDE = (
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include"
)
_CLANG_HOST = _REPO_ROOT / "third_party" / "llvm" / "bin" / "clang++"


def _tcgen05_cuda_flags() -> list[str]:
    flags = [
        "-std=c++20",
        "-gencode=arch=compute_100,code=sm_100",
        "-lineinfo",
        f"-I{_CUTLASS_INCLUDE}",
        f"-I{_LEGACY_CUTLASS_INCLUDE}",
    ]
    if _CLANG_HOST.exists():
        flags.append(f"-ccbin={_CLANG_HOST}")
    return flags


def _load_extension(name: str, sources: Sequence[Path]):
    return load(
        name=name,
        sources=[str(src) for src in sources],
        extra_cuda_cflags=_tcgen05_cuda_flags(),
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


@lru_cache(None)
def load_matmul_tcgen05_module():
    """Compile (if needed) and return the Chapter 10 tcgen05 matmul extension."""
    return _load_extension("ch10_matmul_tcgen05_ext", [_REPO_ROOT / "ch10" / "matmul_tcgen05.cu"])


@lru_cache(None)
def load_tiling_tcgen05_module():
    """Compile (if needed) and return the Chapter 8 tcgen05 tiling extension."""
    return _load_extension("ch8_tiling_tcgen05_ext", [_REPO_ROOT / "ch8" / "tiling_kernels_tcgen05.cu"])


def matmul_tcgen05(a: torch.Tensor, b: torch.Tensor, *, module_name: str = "tcgen05 matmul") -> torch.Tensor:
    """Execute the CUTLASS tcgen05 GEMM after ensuring hardware/toolchain support."""
    ensure_tcgen05_supported(loader=load_matmul_tcgen05_module, module_name=module_name)
    module = load_matmul_tcgen05_module()
    return module.matmul_tcgen05(a, b)


def matmul_tiling_tcgen05(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    module_name: str = "tcgen05 tiling matmul",
) -> torch.Tensor:
    """Execute the CUTLASS tcgen05 tiling GEMM."""
    ensure_tcgen05_supported(loader=load_tiling_tcgen05_module, module_name=module_name)
    module = load_tiling_tcgen05_module()
    return module.matmul_tiling_tcgen05(a, b)


__all__ = [
    "load_matmul_tcgen05_module",
    "load_tiling_tcgen05_module",
    "matmul_tcgen05",
    "matmul_tiling_tcgen05",
]
