"""baseline_cutlass_memory - Baseline GEMM without CUTLASS memory optimization. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch1.workload_config import WORKLOAD


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineCutlassMemoryBenchmark(Benchmark):
    """Baseline: GEMM without CUTLASS memory optimization (standard PyTorch matmul)."""

    def __init__(self):
        self.device = resolve_device()
        self.A_batches = None
        self.B_batches = None
        self.m = 1024
        self.n = 1024
        self.k = 1024
        self.num_steps = max(8, WORKLOAD.performance_microbatches // 4)

    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        self.A_batches = []
        self.B_batches = []
        for _ in range(self.num_steps):
            self.A_batches.append(torch.randn(self.m, self.k, device=self.device, dtype=torch.float32))
            self.B_batches.append(torch.randn(self.k, self.n, device=self.device, dtype=torch.float32))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Standard GEMM without memory optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_cutlass_memory", enable=enable_nvtx):
            acc = 0.0
            for A, B in zip(self.A_batches, self.B_batches):
                acc += torch.matmul(A, B).sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._dummy = acc


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A_batches = None
        self.B_batches = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A_batches is None or self.B_batches is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineCutlassMemoryBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline CUTLASS Memory (Standard): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
