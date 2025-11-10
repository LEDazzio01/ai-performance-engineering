"""baseline_low_ai.py - Low arithmetic intensity baseline (baseline).

Memory-bound kernel with low arithmetic intensity.
Many memory operations relative to compute operations.

Implements Benchmark protocol for harness integration.
"""

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
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineArithmeticIntensityBenchmark(Benchmark):
    """Low arithmetic intensity baseline - memory-bound."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.M = 2048
        self.K = 2048
        self.N = 2048
        self.block_k = 128  # Small tiles -> repeated memory traffic
    
    def setup(self) -> None:
        """Setup: Initialize large tensors."""
        torch.manual_seed(42)
        
        # Allocate matrices for chunked matmul accumulation.
        self.A = torch.randn(self.M, self.K, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.K, self.N, device=self.device, dtype=torch.float32)
        self.C = torch.zeros(self.M, self.N, device=self.device, dtype=torch.float32)

        # Warm up chunked kernel launches.
        self._chunked_matmul()
        torch.cuda.synchronize()

    def _chunked_matmul(self) -> None:
        """Compute C = A @ B using small K-tiles."""
        assert self.A is not None and self.B is not None and self.C is not None
        self.C.zero_()
        for k in range(0, self.K, self.block_k):
            k_end = min(k + self.block_k, self.K)
            a_slice = self.A[:, k:k_end]
            b_slice = self.B[k:k_end, :]
            # Smaller matmuls yield low arithmetic intensity due to repeated reads/writes.
            self.C.addmm_(a_slice, b_slice, beta=1.0, alpha=1.0)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - low arithmetic intensity (memory-bound)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_arithmetic_intensity", enable=enable_nvtx):
            if self.A is None or self.B is None or self.C is None:
                raise RuntimeError("Benchmark not initialized")
            self._chunked_matmul()
            torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=25,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineArithmeticIntensityBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Arithmetic Intensity: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
