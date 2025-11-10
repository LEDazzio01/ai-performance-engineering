"""optimized_cuda_graphs.py - CUDA graph replay (optimized).

Demonstrates CUDA graph capture and replay for reduced launch overhead.
Uses PyTorch CUDA extension for accurate GPU timing with CUDA Events.
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

from ch12.cuda_extensions import load_cuda_graphs_extension


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class OptimizedCudaGraphsBenchmark(Benchmark):
    """CUDA graph replay - reduced launch overhead through graph optimization (uses CUDA extension)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 1 << 18  # Match baseline problem size.
        self.iterations = 200
        self._extension = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self._extension = load_cuda_graphs_extension()
        
        torch.manual_seed(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
        # Warm up graph replay so capture/instantiation happens before measurement.
        self._extension.graph_replay(self.data, 1)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUDA graph replay."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("cuda_graphs", enable=enable_nvtx):
            self._extension.graph_replay(self.data, self.iterations)
            torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
            measurement_timeout_seconds=120,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCudaGraphsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized CUDA Graphs (Graph Replay): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
