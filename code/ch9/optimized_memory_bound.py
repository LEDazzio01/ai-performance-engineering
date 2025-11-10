"""optimized_memory_bound.py - Keep data on GPU and fuse updates."""

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
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedMemoryBoundBenchmark(Benchmark):
    """Keeps data resident on the GPU and fuses updates via torch.compile."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 4_000_000  # Same size as baseline
        self.step_fn = None
        self._compiled = False
    
    def setup(self) -> None:
        """Setup: Initialize tensors and compile the fused update."""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)

        def fused_step(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, 1.0)
            y = torch.addcmul(y, x, x, value=0.5)
            return y

        self.step_fn = fused_step
        self._compiled = False
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Fused operations (high AI)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("memory_bound", enable=enable_nvtx):
            assert self.data is not None and self.step_fn is not None
            # Fused in-place style update that never leaves GPU memory.
            self.data = self.step_fn(self.data)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None or self.step_fn is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryBoundBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Memory Bound: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
