"""baseline_cutlass_memory - Baseline GEMM without CUTLASS memory optimization. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch19.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

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


class BaselineCutlassMemoryBenchmark(Benchmark):
    """Baseline: GEMM without CUTLASS memory optimization (standard PyTorch matmul)."""

    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.output = None
        self.m = 1024
        self.n = 1024
        self.k = 1024
        self.block_size = 128

    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        # Baseline intentionally materializes matmul via many tiny tiles which thrash L2
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.output = torch.zeros(self.m, self.n, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Naive tiled GEMM that replays tiny kernels serially."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_cutlass_memory", enable=enable_nvtx):
            self._blocked_matmul()
            torch.cuda.synchronize()

    def _blocked_matmul(self) -> None:
        """Mimic an unoptimized GEMM that issues many small kernels."""
        self.output.zero_()
        block = self.block_size
        for row in range(0, self.m, block):
            row_end = min(row + block, self.m)
            a_row = self.A[row:row_end]
            for col in range(0, self.n, block):
                col_end = min(col + block, self.n)
                tile = self.output[row:row_end, col:col_end]
                for depth in range(0, self.k, block):
                    depth_end = min(depth + block, self.k)
                    prod = torch.mm(
                        a_row[:, depth:depth_end],
                        self.B[depth:depth_end, col:col_end],
                    )
                    tile.add_(prod)

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.output is None:
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
