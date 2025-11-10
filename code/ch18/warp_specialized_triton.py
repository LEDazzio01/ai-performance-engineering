"""Triton warp specialization kernel for Chapter 18.

Demonstrates warp specialization using Triton's warp_specialize=True feature.
Based on Chapter 18's attention context - warp specialization for attention heads.
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
def warp_specialized_triton_kernel_ch18(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Warp-specialized Triton kernel for Chapter 18 attention context."""

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

        q_data = tl.load(q_ptr + offsets, mask=mask, other=0.0)
        k_data = tl.load(k_ptr + offsets, mask=mask, other=0.0)
        v_data = tl.load(v_ptr + offsets, mask=mask, other=0.0)

        scores = (q_data * k_data) * SCALE
        scores = tl.maximum(scores, 0.0)
        result = scores * v_data
        tl.store(output_ptr + offsets, result, mask=mask)


def warp_specialized_triton_forward_ch18(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """Warp-specialized forward pass using Triton for Chapter 18."""
    output = torch.empty_like(q)
    n_elements = q.numel()
    
    BLOCK_SIZE = 256
    PIPELINE_STAGES = 4
    TILE_SIZE = BLOCK_SIZE * PIPELINE_STAGES
    SCALE = 1.0 / 8.0  # match sqrt(head_dim) for head_dim=64
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["TILE_SIZE"]),)
    
    warp_specialized_triton_kernel_ch18[grid](
        q,
        k,
        v,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
        PIPELINE_STAGES=PIPELINE_STAGES,
        SCALE=SCALE,
        num_warps=8,
        num_stages=PIPELINE_STAGES,
    )
    
    return output
