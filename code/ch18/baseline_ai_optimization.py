"""Naive attention flow without any AI-driven optimization heuristics."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch18.workload_config import WORKLOAD, is_smoke_test

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class NaiveAttentionBlock(nn.Module):
    """Reference attention layer that iterates head-by-head."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        head_outputs: list[torch.Tensor] = []
        for head_idx in range(self.num_heads):
            q_h = q[:, head_idx]
            k_h = k[:, head_idx]
            v_h = v[:, head_idx]
            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            head_outputs.append(torch.matmul(attn, v_h))

        stacked = torch.stack(head_outputs, dim=2).reshape(batch, seq, self.hidden_dim)
        return self.proj(stacked)


class BaselineAiOptimizationBenchmark(Benchmark):
    """Sequential attention processing with fixed heuristics."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.hidden_dim = self.workload.attention_hidden_dim
        self.num_heads = self.workload.attention_num_heads
        self.batch_size = self.workload.attention_batch_size
        self.sequence_length = self.workload.seq_len(self.smoke_test)
        self.micro_batches = self.workload.micro_batches_for_mode(self.smoke_test)

        self.block: Optional[NaiveAttentionBlock] = None
        self.token_cache: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.block = NaiveAttentionBlock(self.hidden_dim, self.num_heads).to(self.device).half().eval()
        self.token_cache = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.block is not None
        assert self.token_cache is not None

        with nvtx_range("baseline_ai_optimization", enable=enable_nvtx):
            for micro in range(self.micro_batches):
                rolled = torch.roll(self.token_cache, shifts=micro * 32, dims=1).contiguous()
                _ = self.block(rolled)
                torch.cuda.synchronize()

    def teardown(self) -> None:
        self.block = None
        self.token_cache = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=2,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=120,
        )

    def validate_result(self) -> Optional[str]:
        if self.block is None:
            return "Attention block not initialized"
        if self.token_cache is None:
            return "Input cache not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineAiOptimizationBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(BaselineAiOptimizationBenchmark())
    print(f"Baseline AI optimization mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
