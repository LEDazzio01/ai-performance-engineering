"""baseline_cutlass.py - Baseline GEMM without CUTLASS optimization.

Demonstrates standard GEMM without CUTLASS library optimization.
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
        raise RuntimeError("CUDA required for ch14")
    return torch.device("cuda")


class BaselineCutlassBenchmark(Benchmark):
    """Baseline: GEMM without CUTLASS optimization (standard PyTorch matmul)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        # Match optimized matrix size for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
        self.block_k = 128
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        # Baseline: Standard PyTorch matmul (FP16) without CUTLASS optimization
        # Using FP16 to match optimized version for fair comparison
        # Disable TF32 to use standard GEMM kernels (not CUTLASS-optimized)
        # Match optimized version backend settings for fair comparison
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("high")
        # Match optimized version cuDNN settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # Match optimized version dtype for fair comparison
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.C = torch.zeros(self.m, self.n, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()

    def _naive_matmul(self) -> torch.Tensor:
        """Compute C = A @ B using many small GEMMs (poor locality)."""
        assert self.A is not None and self.B is not None and self.C is not None
        self.C.zero_()
        for k in range(0, self.k, self.block_k):
            k_end = min(k + self.block_k, self.k)
            a_slice = self.A[:, k:k_end]
            b_slice = self.B[k:k_end, :]
            self.C.add_(torch.matmul(a_slice, b_slice))
        return self.C
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard GEMM."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_cutlass", enable=enable_nvtx):
            if self.A is None or self.B is None or self.C is None:
                raise RuntimeError("Benchmark not initialized")
            # Baseline: naive blocked matmul built from many GEMM calls.
            _ = self._naive_matmul()
            torch.cuda.synchronize(self.device)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.C = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
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
    return BaselineCutlassBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline CUTLASS (Standard): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
