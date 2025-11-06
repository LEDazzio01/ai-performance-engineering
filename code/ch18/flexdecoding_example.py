from __future__ import annotations
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

# Environment (Oct-2025): CUDA 13.x r580+, torch 2.9.0+cu130, triton 3.5.0, optional TE 2.8+
"""FlexDecoding showcase aligned with PyTorch 2.9 (CUDA 13.0)."""

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    pass
except Exception:
    class arch_config:  # type: ignore[override]
        @staticmethod
        def is_blackwell() -> bool:
            import torch
            if not torch.cuda.is_available():
                return False
            major, minor = torch.cuda.get_device_capability()
            return major >= 12


import math
import time
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import flex_attention
    HAS_FLEX = True
except (ImportError, AttributeError):
    HAS_FLEX = False

QUICK_MODE = any(
    os.getenv(flag, "0") == "1"
    for flag in ("QUICK_PROFILE", "BENCHMARK_QUICK", "RUN_ALL_CHAPTERS")
)
DEFAULT_COMPILE_MODE = "reduce-overhead" if QUICK_MODE else "default"
COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", DEFAULT_COMPILE_MODE)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(label: str, fn, iters: int) -> float:
    _sync()
    start = time.perf_counter()
    if QUICK_MODE:
        iters = min(iters, 5)
    with torch.inference_mode():
        for _ in range(iters):
            fn()
    _sync()
    elapsed = (time.perf_counter() - start) * 1_000 / iters
    print(f"{label:<28}: {elapsed:7.3f} ms")
    return elapsed


def _score_mod_causal(offset: torch.Tensor):
    def score_mod(score, _b, _h, q_idx, kv_idx):
        q = q_idx + offset
        return torch.where(q >= kv_idx, score, torch.neg(torch.inf_like(score)))

    return score_mod


@dataclass
class FlexDecodingConfig:
    dim: int = 256
    heads: int = 4
    max_seq_len: int = 1024
    window: int = 128


class FlexDecodingModule(torch.nn.Module):
    def __init__(self, cfg: FlexDecodingConfig):
        super().__init__()
        assert cfg.dim % cfg.heads == 0
        self.cfg = cfg
        self.head_dim = cfg.dim // cfg.heads

        self.q_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.o_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.register_buffer("k_cache", torch.zeros(1, cfg.max_seq_len, cfg.heads, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, cfg.max_seq_len, cfg.heads, self.head_dim))
        self.register_buffer("offset", torch.zeros(1, dtype=torch.long))

        self.prefill_impl = None
        self.decode_impl = None
        self.flex_enabled = HAS_FLEX
        assert torch.cuda.is_available(), "CUDA required for FlexDecoding demo"
        major, minor = torch.cuda.get_device_capability()
        assert major >= 12, f"Blackwell expected (sm_120); got sm_{major}{minor}"
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        self.compile_mode = COMPILE_MODE

    def _compile(self, pattern: str = "causal") -> None:
        device = next(self.parameters()).device
        head_dim = self.head_dim
        heads = self.cfg.heads

        q_prefill = torch.randn(1, heads, 256, head_dim, device=device)
        kv_prefill = torch.randn_like(q_prefill)
        q_decode = torch.randn(1, heads, 1, head_dim, device=device)
        compile_kwargs = {"mode": self.compile_mode, "dynamic": None}

        def configure_sdpa_fallback() -> None:
            def sdpa(q, k, v):
                # Inputs expected as [batch, seq, heads, head_dim]
                qh = q.transpose(1, 2)
                kh = k.transpose(1, 2)
                vh = v.transpose(1, 2)
                scale = 1.0 / math.sqrt(float(qh.size(-1)))
                attn = torch.matmul(qh, kh.transpose(-2, -1)) * scale
                seq_q = attn.size(-2)
                seq_k = attn.size(-1)
                q_positions = torch.arange(seq_q, device=attn.device).unsqueeze(-1)
                k_positions = torch.arange(seq_k, device=attn.device).unsqueeze(0)
                causal = (q_positions >= k_positions).to(attn.dtype)
                attn = attn + (causal - 1.0) * 1e9  # large negative for masked entries
                probs = torch.softmax(attn, dim=-1)
                out = torch.matmul(probs, vh)
                return out.transpose(1, 2)

            try:
                compiled_prefill = torch.compile(sdpa, **compile_kwargs)
                compiled_decode = torch.compile(sdpa, **compile_kwargs)

                q_prefill_eager = q_prefill.transpose(1, 2)
                kv_prefill_eager = kv_prefill.transpose(1, 2)
                compiled_prefill(q_prefill_eager, kv_prefill_eager, kv_prefill_eager)
                compiled_decode(q_prefill_eager[:, :1], kv_prefill_eager, kv_prefill_eager)

                self.prefill_impl = compiled_prefill
                self.decode_impl = compiled_decode
            except Exception as exc:
                warnings.warn(
                    f"torch.compile failed for SDPA fallback; using eager execution: {exc}",
                    stacklevel=2,
                )
                self.prefill_impl = sdpa
                self.decode_impl = sdpa
            self.flex_enabled = False

        if HAS_FLEX:
            score_mod = _score_mod_causal(self.offset)

            def prefill(q, k, v):
                return flex_attention.flex_attention(q, k, v, score_mod=score_mod)

            def decode(q, k, v):
                return flex_attention.flex_attention(q, k, v, score_mod=score_mod)

            try:
                self.prefill_impl = torch.compile(prefill, **compile_kwargs)
                self.decode_impl = torch.compile(decode, **compile_kwargs)

                self.prefill_impl(q_prefill, kv_prefill, kv_prefill)
                self.decode_impl(q_decode, kv_prefill, kv_prefill)
                self.flex_enabled = True
            except Exception as exc:
                warnings.warn(
                    f"FlexAttention torch.compile failed; falling back to eager mode: {exc}",
                    stacklevel=2,
                )
                configure_sdpa_fallback()
        else:
            configure_sdpa_fallback()

    def ensure_compiled(self) -> None:
        if self.prefill_impl is None or self.decode_impl is None:
            self._compile()

    def clear_cache(self, batch: int) -> None:
        if self.k_cache.shape[0] != batch:
            device = self.k_cache.device
            self.k_cache = torch.zeros(batch, self.cfg.max_seq_len, self.cfg.heads, self.head_dim, device=device)
            self.v_cache = self.k_cache.clone()
        else:
            self.k_cache.zero_()
            self.v_cache.zero_()

    def prefill(self, tokens: torch.Tensor, past: int = 0) -> torch.Tensor:
        batch, seqlen, _ = tokens.shape
        self.ensure_compiled()
        if self.k_cache.shape[0] != batch:
            self.clear_cache(batch)

        q = self.q_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)
        k = self.k_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)
        v = self.v_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)

        self.k_cache[:, past:past + seqlen] = k
        self.v_cache[:, past:past + seqlen] = v

        if self.flex_enabled:
            out = self.prefill_impl(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.prefill_impl(q, k, v)
        return self.o_proj(out.reshape(batch, seqlen, self.cfg.dim))

    def decode(self, token: torch.Tensor, position: int) -> torch.Tensor:
        batch, _, _ = token.shape
        self.ensure_compiled()

        q = self.q_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)
        k = self.k_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)
        v = self.v_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)

        self.k_cache[:, position:position + 1] = k
        self.v_cache[:, position:position + 1] = v
        past_k = self.k_cache[:, :position + 1]
        past_v = self.v_cache[:, :position + 1]

        self.offset.fill_(position)

        if self.flex_enabled:
            out = self.decode_impl(q.transpose(1, 2), past_k.transpose(1, 2), past_v.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.decode_impl(q, past_k, past_v)
        return self.o_proj(out.reshape(batch, 1, self.cfg.dim))


def jagged_batch(model: FlexDecodingModule, sequences: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for seq in sequences:
            model.clear_cache(seq.shape[0])
            pref = model.prefill(seq)
            cur = pref[:, -1:, :]
            tokens = [cur]
            for step in range(3):
                nxt = model.decode(cur, seq.shape[1] + step)
                tokens.append(nxt)
                cur = nxt
            outputs.append(torch.cat(tokens, dim=1))
    return outputs


def benchmark(model: FlexDecodingModule) -> None:
    device = _device()
    torch.manual_seed(0)

    batch = 4
    seq_len = 256
    tokens = torch.randn(batch, seq_len, model.cfg.dim, device=device)
    model.ensure_compiled()

    print("\nPrefill vs decode timings")
    _benchmark("Prefill", lambda: model.prefill(tokens), iters=3)
    single = torch.randn(batch, 1, model.cfg.dim, device=device)
    _benchmark("Decode", lambda: model.decode(single, seq_len), iters=20)


def paged_attention_demo() -> None:
    print("\nPagedAttention-style block mapping")
    batch, heads, head_dim = 2, 4, 32
    block, blocks = 8, 16
    logical = torch.randn(batch, blocks * block, heads, head_dim)
    physical = torch.zeros(blocks, block, heads, head_dim)
    table = torch.randint(0, blocks, (batch, blocks))
    for b in range(batch):
        for blk in range(blocks):
            src = logical[b, blk * block : (blk + 1) * block]
            dst = table[b, blk]
            physical[dst].copy_(src)
    print(f"Logical {logical.shape} -> Physical {physical.shape}")


def main() -> None:
    device = _device()
    print("FlexDecoding example (PyTorch 2.9 / CUDA 13.0)")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    cfg = FlexDecodingConfig()
    model = FlexDecodingModule(cfg).to(device)
    model.ensure_compiled()

    with torch.inference_mode():
        prompt = torch.randn(1, 32, cfg.dim, device=device)
        print("\nPrefill output shape", model.prefill(prompt).shape)
        token = torch.randn(1, 1, cfg.dim, device=device)
        print("Decode output shape", model.decode(token, 32).shape)

        sequences = [
            torch.randn(1, 16, cfg.dim, device=device),
            torch.randn(1, 32, cfg.dim, device=device),
            torch.randn(1, 64, cfg.dim, device=device),
        ]
    outs = jagged_batch(model, sequences)
    for idx, out in enumerate(outs):
        print(f"Sequence {idx}: {out.shape[1]} tokens emitted")

    if torch.cuda.is_available():
        benchmark(model)

    paged_attention_demo()

    print("\nKey takeaways:")
    print("- torch.compile can specialize prefill vs decode paths.")
    print("- FlexAttention (when available) supplies custom score mods.")
    print("- Jagged batches illustrate variable sequence handling.")
    print("- Block remapping mirrors PagedAttention KV packing.")


if __name__ == "__main__":
    main()
