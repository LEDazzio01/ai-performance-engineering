"""Baseline flex attention – naive per-head computation without fusion."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import math
import torch
from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch14")
    return torch.device("cuda")


class BaselineFlexAttentionBenchmark(Benchmark):
    """Baseline: Naive attention that iterates per head without fusion."""

    def __init__(self):
        self.device = resolve_device()
        self.q = None
        self.k = None
        self.v = None
        self.num_heads = 16
        self.head_dim = 64
        self.seq_len = 1024
        self._last = 0.0
        self.repeat_passes = 8

    def setup(self) -> None:
        """Setup: materialize query/key/value tensors."""
        torch.manual_seed(42)
        shape = (self.seq_len, self.num_heads, self.head_dim)
        self.q = torch.randn(shape, device=self.device, dtype=torch.float32)
        self.k = torch.randn(shape, device=self.device, dtype=torch.float32)
        self.v = torch.randn(shape, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: per-head attention computed serially."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_flex_attention", enable=enable_nvtx):
            if self.q is None or self.k is None or self.v is None:
                raise RuntimeError("Tensors not initialized")
            scale = 1.0 / math.sqrt(self.head_dim)
            outputs = []
            for _ in range(self.repeat_passes):
                for head in range(self.num_heads):
                    qh = self.q[:, head, :]
                    kh = self.k[:, head, :]
                    vh = self.v[:, head, :]
                    scores = torch.matmul(qh, kh.transpose(0, 1)) * scale
                    attn = torch.softmax(scores, dim=-1)
                    outputs.append(torch.matmul(attn, vh))
            stacked = torch.stack(outputs, dim=1)
            self._last = float(stacked.sum())
            torch.cuda.synchronize(self.device)


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.q = None
        self.k = None
        self.v = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.q is None or self.k is None or self.v is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineFlexAttentionBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Flex Attention (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
