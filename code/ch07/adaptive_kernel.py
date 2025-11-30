"""Shared Triton kernel for adaptive tiling benchmarks."""

from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _adaptive_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply moderately expensive element-wise math to keep GPU busy
    values = values * 1.61803399 + tl.sin(values) * 0.5
    tl.store(y_ptr + offsets, values, mask=mask)


def run_kernel(x: torch.Tensor, y: torch.Tensor, block_size: int) -> None:
    """Launch the adaptive Triton kernel with the requested tile size."""
    assert x.is_cuda and y.is_cuda
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    _adaptive_kernel[grid](x, y, x.numel(), BLOCK_SIZE=block_size)
