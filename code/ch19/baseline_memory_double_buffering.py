"""baseline_memory_double_buffering.py - Single-stream baseline."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class MemoryDoubleBufferingBenchmark(Benchmark):
    """Baseline: single stream, single buffer (no overlap)."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 1024

    def setup(self) -> None:
        """Setup: Initialize single-GPU tensors."""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        ).to(self.device).half().eval()
        self.buffer = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.output = torch.empty_like(self.buffer)
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            self.output.copy_(self.buffer, non_blocking=True)
        self.stream.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Single-GPU stream-ordered operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        assert self.model is not None and self.buffer is not None and self.stream is not None
        with nvtx_range("baseline_memory_double_buffering", enable=enable_nvtx):
            with torch.no_grad():
                with torch.cuda.stream(self.stream):
                    self.output = self.model(self.buffer)
            self.stream.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.buffer = None
        self.output = None
        self.stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return MemoryDoubleBufferingBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Memory double buffering (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
