"""Utility loaders for labs.moe_cuda CUDA decode kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from common.python.extension_loader_template import load_cuda_extension

SUITE_DIR = Path(__file__).parent
KERNEL_DIR = SUITE_DIR / "kernels"


@lru_cache(maxsize=None)
def _load_baseline_module():
    return load_cuda_extension(
        extension_name="moe_cuda_decode_baseline",
        cuda_source_file=str(KERNEL_DIR / "baseline_decode_kernel.cu"),
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
    )


@lru_cache(maxsize=None)
def _load_optimized_module():
    return load_cuda_extension(
        extension_name="moe_cuda_decode_optimized",
        cuda_source_file=str(KERNEL_DIR / "optimized_decode_kernel.cu"),
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
        extra_ldflags=["-lcuda"],
    )


def run_baseline_kernel(input_tensor, output_tensor) -> None:
    """Run the baseline CUDA decode kernel."""
    module = _load_baseline_module()
    module.run_baseline(input_tensor, output_tensor)


def run_optimized_kernel(input_tensor, output_tensor) -> None:
    """Run the optimized CUDA decode kernel (TMA + overlap)."""
    module = _load_optimized_module()
    module.run_optimized(input_tensor, output_tensor)


def optimized_kernel_supported(rows: int, cols: int) -> bool:
    """Return True if the optimized kernel can run on this GPU for given shape."""
    module = _load_optimized_module()
    # Leading dimension equals number of columns for contiguous tensors
    return bool(module.supports_tma(rows, cols, cols))
