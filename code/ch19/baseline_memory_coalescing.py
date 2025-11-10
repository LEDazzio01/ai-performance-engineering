"""Baseline stream-ordered single-GPU (no distributed)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
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


class MemoryCoalescingBenchmark(Benchmark):
    """Baseline: Single-GPU stream-ordered (no distributed computing)."""

    def __init__(self):
        self.device = resolve_device()
        self.input_matrix = None
        self.output_matrix = None
        self.shuffle = None
        self.chunk = 32
        self.rows = 2048
        self.cols = 1024

    def setup(self) -> None:
        """Setup: Initialize tensors with poor memory layout."""
        torch.manual_seed(42)
        # Baseline: intentionally touch columns in random order which defeats coalescing
        self.input_matrix = torch.randn(
            self.rows, self.cols, device=self.device, dtype=torch.float32
        )
        self.output_matrix = torch.empty_like(self.input_matrix)
        self.shuffle = torch.randperm(self.cols, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Column scatter/gather with non-coalesced memory."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_memory_coalescing", enable=enable_nvtx):
            matrix = self.input_matrix
            scale = 1.0003
            bias = 0.01
            for start in range(0, self.cols, self.chunk):
                cols = self.shuffle[start:start + self.chunk]
                gathered = matrix[:, cols]
                transformed = gathered * scale + bias
                self.output_matrix[:, cols] = transformed
            torch.cuda.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input_matrix = None
        self.output_matrix = None
        self.shuffle = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output_matrix is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return MemoryCoalescingBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Memory coalescing (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
