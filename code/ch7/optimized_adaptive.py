"""optimized_adaptive.py - Adaptive tile size (optimized).

Demonstrates adaptive optimization with dynamic tile sizing based on workload.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import arch_config  # noqa: F401
except ImportError:
    pass
import torch

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

from ch7.adaptive_kernel import run_kernel


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")


class OptimizedAdaptiveBenchmark(Benchmark):
    """Adaptive tile size - runtime optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 10_000_000
        self.tile_size = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and determine optimal tile size."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        
        # Adaptive optimization: benchmark real tile sizes on the GPU
        candidate_tiles = [64, 128, 256, 512, 1024]
        self.tile_size = self._autotune_tile_size(candidate_tiles)
        torch.cuda.synchronize()

    def _autotune_tile_size(self, candidates):
        """Measure each tile size using CUDA events and pick the fastest."""
        timings = {}
        for tile in candidates:
            run_kernel(self.input, self.output, tile)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            total_ms = 0.0
            trials = 3
            for _ in range(trials):
                start.record()
                run_kernel(self.input, self.output, tile)
                end.record()
                torch.cuda.synchronize()
                total_ms += start.elapsed_time(end)
            timings[tile] = total_ms / trials
        best_tile = min(timings, key=timings.get)
        return best_tile
    
    def benchmark_fn(self) -> None:
        """Benchmark: Adaptive tile size processing."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("adaptive", enable=enable_nvtx):
            run_kernel(self.input, self.output, self.tile_size)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        if self.input is None:
            return "Input tensor not initialized"
        if self.output.shape[0] != self.N:
            return f"Output shape mismatch: expected {self.N}, got {self.output.shape[0]}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAdaptiveBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Adaptive (Dynamic Tile Size): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
