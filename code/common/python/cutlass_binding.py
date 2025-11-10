"""CUTLASS PyTorch binding for Chapter 14."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from common.python.extension_loader_template import load_cuda_extension


def _detect_cutlass_include() -> Path:
    """Return the CUTLASS include directory shipped with nvidia-cutlass-dsl."""
    try:
        import cutlass_library
    except ImportError as exc:
        raise RuntimeError(
            "CUTLASS Python package (cutlass_library) is missing. "
            "Install nvidia-cutlass-dsl>=4.2 to build CUTLASS extensions."
        ) from exc
    
    include_dir = Path(cutlass_library.__file__).resolve().parent / "source" / "include"
    if not include_dir.exists():
        raise RuntimeError(
            f"CUTLASS include directory not found at {include_dir}. "
            "Ensure nvidia-cutlass-dsl is properly installed."
        )
    return include_dir


@lru_cache()
def _load_cutlass_module(verbose: bool = False):
    """Compile and cache the CUTLASS GEMM extension."""
    repo_root = Path(__file__).resolve().parents[2]
    cuda_source = repo_root / "common" / "cuda" / "cutlass_gemm_extension.cu"
    include_dir = _detect_cutlass_include()
    extra_flags = ["-O3", "--use_fast_math", "-std=c++17"]
    return load_cuda_extension(
        extension_name="cutlass_gemm_ext",
        cuda_source_file=str(cuda_source),
        include_dirs=[include_dir],
        extra_cuda_cflags=extra_flags,
        verbose=verbose,
    )


def cutlass_gemm_fp16(a: torch.Tensor, b: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """Invoke the CUTLASS GEMM kernel from Python."""
    module = _load_cutlass_module(verbose=verbose)
    return module.cutlass_gemm_fp16(a, b)
