"""Triton warp specialization kernel for Chapter 13.

Demonstrates warp specialization using Triton's warp_specialize=True feature.
Based on Chapter 13's training context - warp specialization for gradient computation.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python import triton_compat  # noqa: F401

import torch
import triton
import triton.language as tl


@triton.jit
def warp_specialized_triton_kernel_ch13(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Warp-specialized Triton kernel for Chapter 13 training context."""

    pid = tl.program_id(axis=0)
    tile_start = pid * TILE_SIZE

    for chunk in tl.range(
        0,
        TILE_SIZE,
        BLOCK_SIZE,
        num_stages=PIPELINE_STAGES,
        warp_specialize=True,
    ):
        offsets = tile_start + chunk + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        weight_data = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        result = tl.maximum(input_data * weight_data, 0.0)
        tl.store(output_ptr + offsets, result, mask=mask)


def warp_specialized_triton_forward_ch13(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Warp-specialized forward pass using Triton for Chapter 13."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 256
    PIPELINE_STAGES = 4
    TILE_SIZE = BLOCK_SIZE * PIPELINE_STAGES  # 1024 elements per tile
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["TILE_SIZE"]),)
    
    warp_specialized_triton_kernel_ch13[grid](
        x,
        weight,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
        PIPELINE_STAGES=PIPELINE_STAGES,
        num_warps=8,
        num_stages=PIPELINE_STAGES,
    )
    
    return output
