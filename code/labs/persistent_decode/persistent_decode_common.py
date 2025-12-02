"""Common helpers for persistent decode lab."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for persistent decode lab")
    return torch.device("cuda")


def get_stream_priorities() -> Tuple[int, int]:
    """Return (low_priority, high_priority) for the current CUDA device."""
    low, high = torch.cuda.Stream.priority_range()
    return low, high


@dataclass
class DecodeProfile:
    tier: str
    block_k: int
    num_programs: int


@dataclass
class DecodeOptions:
    tier: str = "medium"  # small | medium | large
    quantization: str = "fp32"  # fp32 | fp16 | int4
    block_k: Optional[int] = None
    num_programs: Optional[int] = None


_PROFILE_BY_TIER = {
    "small": DecodeProfile(tier="small", block_k=32, num_programs=4),
    "medium": DecodeProfile(tier="medium", block_k=64, num_programs=8),
    "large": DecodeProfile(tier="large", block_k=64, num_programs=12),
}

_OPTIONS = DecodeOptions()


def set_decode_options(opts: DecodeOptions) -> None:
    """Update global decode options (set via CLI flags in right-sized benchmarks)."""
    global _OPTIONS
    _OPTIONS = opts


def get_decode_options() -> DecodeOptions:
    return _OPTIONS


def get_decode_profile() -> DecodeProfile:
    base = _PROFILE_BY_TIER.get(_OPTIONS.tier, _PROFILE_BY_TIER["medium"])
    prof = replace(base)
    if _OPTIONS.block_k:
        prof.block_k = _OPTIONS.block_k
    if _OPTIONS.num_programs:
        prof.num_programs = _OPTIONS.num_programs
    return prof


def resolve_shapes() -> Tuple[int, int, int]:
    """
    Resolve (batch, seq_len, head_dim) with tier-aware defaults.

    - small: lighter shapes to mimic a smaller decode slice
    - medium: default shapes (8 x 32) for continuity with earlier labs
    - large: heavier batch to mimic a bigger slice
    """
    if _OPTIONS.tier == "small":
        return 4, 32, 64
    if _OPTIONS.tier == "large":
        return 12, 64, 64
    return 8, 32, 64


@dataclass
class DecodeInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    work_seq_ids: torch.Tensor
    work_steps: torch.Tensor
    work_counter: torch.Tensor


def build_inputs(device: torch.device) -> DecodeInputs:
    batch, seq_len, head_dim = resolve_shapes()
    torch.manual_seed(0)

    quant = _OPTIONS.quantization.lower()
    dtype = torch.float32 if quant == "fp32" else torch.float16

    q = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if quant == "int4":
        q = _fake_int4(q)
        k = _fake_int4(k)
        v = _fake_int4(v)

    out = torch.zeros_like(q)

    work_seq_ids = torch.arange(batch, device=device, dtype=torch.int32)
    work_steps = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    work_counter = torch.zeros(1, device=device, dtype=torch.int32)

    return DecodeInputs(
        q=q,
        k=k,
        v=v,
        out=out,
        work_seq_ids=work_seq_ids,
        work_steps=work_steps,
        work_counter=work_counter,
    )


def tokens_per_iteration() -> float:
    batch, seq_len, _ = resolve_shapes()
    return float(batch * seq_len)


def _fake_int4(t: torch.Tensor) -> torch.Tensor:
    """Toy fake-quantization to simulate INT4 decode pools."""
    max_abs = t.abs().amax()
    if max_abs == 0:
        return t.half()
    scale = max_abs / 7.0
    q = torch.round(t / scale).clamp(-8, 7)
    return (q * scale).half()
