"""baseline_compute_bound.py - Baseline compute-bound kernel (unfused operations).

Executes multiple element-wise math kernels sequentially, reloading data from
global memory between steps. This keeps arithmetic intensity high but leaves
opportunities for fusion exploited by the optimized version.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class BaselineComputeBoundBenchmark(Benchmark):
    """Baseline compute-bound implementation using separate kernels per op."""

    def __init__(self) -> None:
        self.device = resolve_device()
        self.data: Optional[torch.Tensor] = None
        self.N = 10_000_000  # 10M elements to keep kernels busy

    def setup(self) -> None:
        """Allocate input tensor on the GPU."""
        torch.manual_seed(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Run sequential element-wise ops (no fusion, more memory traffic)."""
        if self.data is None:
            raise RuntimeError("CUDA tensors not initialized")

        # Use conditional NVTX ranges - only enabled when profiling


        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled


        config = self.get_config()


        enable_nvtx = get_nvtx_enabled(config) if config else False



        with nvtx_range("baseline_compute_bound", enable=enable_nvtx):
            # Each step writes intermediates back to memory which reduces reuse.
            sin_out = torch.sin(self.data)
            cos_out = torch.cos(self.data)
            product = sin_out * cos_out
            squared = product * product
            sqrt_term = torch.sqrt(torch.abs(product))
            combined = squared + sqrt_term
            self.data = combined * 0.90 + torch.exp(product * 0.001)


    def teardown(self) -> None:
        """Release GPU tensors."""
        self.data = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Use a modest iteration count to keep runtime manageable."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def validate_result(self) -> Optional[str]:
        """Ensure tensor shape and values remain valid."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory hook used by discover_benchmarks()."""
    return BaselineComputeBoundBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Compute Bound: {result.mean_ms:.3f} ms")

