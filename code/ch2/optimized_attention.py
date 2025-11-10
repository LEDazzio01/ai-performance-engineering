"""Optimized attention benchmark for Chapter 2.

Highlights the benefit of running FlashAttention (PyTorch SDP backend) in FP16
with TF32 enabled on Grace-Blackwell C2C links. Compared to the baseline, this
path keeps everything on tensor cores and avoids materializing the full
attention matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2 attention example")
    return torch.device("cuda")


class FlashAttentionBlock(nn.Module):
    """Uses scaled_dot_product_attention which routes to FlashAttention."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq, self.hidden_dim)
        return self.out_proj(ctx)


class OptimizedAttentionBenchmark(Benchmark):
    """FlashAttention in FP16 with TF32-friendly matmul settings."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self._compiled = False

    def setup(self) -> None:
        torch.manual_seed(42)
        enable_tf32()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        model = FlashAttentionBlock(hidden_dim=512, num_heads=8).to(self.device).half().eval()
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            try:
                model = compile_fn(model, mode="reduce-overhead", fullgraph=True)
                self._compiled = True
            except Exception:
                self._compiled = False
        else:
            self._compiled = False
        self.model = model

        seq_len = 512
        self.inputs = torch.randn(4, seq_len, 512, device=self.device, dtype=torch.float16).contiguous()

        # Warm up Lt heuristics and FlashAttention tile selection
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(self.inputs)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None

        with nvtx_range("optimized_attention", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=3)

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model or inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedAttentionBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=2),
    )
    result = harness.benchmark(get_benchmark())
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized attention latency: {timing:.3f} ms")
