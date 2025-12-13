"""Shared helpers for FlexAttention lab benchmarks.

Extracts the flex_attention DSL pieces from the Colfax FlexAttention guide
into utilities that both baseline and optimized lab variants can reuse.

Key knobs exposed by the lab:
 - block_size: tile size passed into create_block_mask (how coarse sparse blocks are)
 - doc_span: tokens per “document” that define boundaries for the mask
 - seq_len/head_dim/heads: shape of the synthetic qkv tensors
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


@dataclass
class FlexAttentionInputs:
    """Container for tensors needed by the lab benchmarks."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_mask: BlockMask
    rel_bias: torch.Tensor


def resolve_device() -> torch.device:
    """Require CUDA for flex attention benchmarks."""
    if not torch.cuda.is_available():
        raise RuntimeError("FlexAttention labs require CUDA")
    return torch.device("cuda")


def _make_doc_ids(batch: int, heads: int, seq_len: int, doc_span: int, device: torch.device) -> torch.Tensor:
    """Assign a document id per token to gate cross-document attention."""
    doc_ids_1d = (torch.arange(seq_len, device=device) // doc_span).to(torch.int32)
    doc_ids = doc_ids_1d.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1).contiguous()
    return doc_ids


def build_flex_attention_inputs(
    *,
    batch: int = 2,
    heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    doc_span: int = 256,
    block_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device,
    use_vmap: bool | None = None,
) -> FlexAttentionInputs:
    """Create qkv tensors, a block mask, and a relative bias table."""
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    doc_ids = _make_doc_ids(batch, heads, seq_len, doc_span, device=device)

    def document_mask(b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        # Return a tensor predicate to keep vmap happy (avoid Tensor->bool control flow).
        return torch.eq(doc_ids[b, h, q_idx], doc_ids[b, h, kv_idx])

    # FlexAttention API surface changed: newer builds expose a use_vmap flag while older
    # (e.g., packaged with this environment) do not. Only pass it when supported to avoid
    # TypeError on older builds.
    mask_kwargs = {"device": device, "BLOCK_SIZE": block_size}
    if "use_vmap" in inspect.signature(create_block_mask).parameters:
        mask_kwargs["use_vmap"] = True if use_vmap is None else use_vmap

    block_mask = create_block_mask(
        document_mask,
        batch,
        heads,
        seq_len,
        seq_len,
        **mask_kwargs,
    )

    rel_positions = torch.arange(seq_len, device=device)
    # Gently decaying bias highlights the score_mod hook from the blog.
    rel_bias_1d = torch.exp(-rel_positions / float(doc_span))
    rel_bias = rel_bias_1d.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1).contiguous()

    return FlexAttentionInputs(q=q, k=k, v=v, block_mask=block_mask, rel_bias=rel_bias)


def build_qkv_inputs(
    *,
    batch: int = 2,
    heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create only qkv tensors for cases where custom masks are not needed."""
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def make_relative_bias_score_mod(rel_bias: torch.Tensor) -> Callable:
    """Return a score_mod closure that adds a distance-based bias."""

    max_rel = rel_bias.size(-1) - 1

    def score_mod(
        score: torch.Tensor,
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        distance = (q_idx - kv_idx).abs().clamp_max(max_rel)
        bias = rel_bias[batch_idx, head_idx, distance]
        return score + bias

    return score_mod
