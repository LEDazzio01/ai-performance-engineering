"""optimized_hbm.py - Optimized HBM memory access with CUDA graphs in kernel launches context.

Demonstrates HBM optimization with CUDA graphs to reduce kernel launch overhead.
HBM: Optimizes memory access patterns for HBM high bandwidth.
Uses CUDA graphs to capture and replay HBM-optimized operations efficiently.
Implements Benchmark protocol for harness integration.
"""

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
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")

class OptimizedHbmBenchmark(Benchmark):
    """Optimized: HBM memory optimization with CUDA graphs.
    
    HBM: Optimizes memory access patterns for HBM high bandwidth.
    Uses CUDA graphs to capture and replay HBM-optimized operations efficiently.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.device_input = None
        self.graph_input = None
        self.graph_output = None
        self.graph = None
        self.batch = 2048
        self._last_total = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model with HBM optimization and CUDA graphs."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: HBM memory optimization with CUDA graphs
        # HBM: optimizes for high bandwidth memory
        # CUDA graphs: capture kernels to reduce launch overhead
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.device_input = torch.randn(self.batch, 1024, device=self.device)
        
        for _ in range(2):
            with torch.no_grad():
                _ = self.model(self.device_input)
        torch.cuda.synchronize()
        
        self.graph_input = self.device_input.clone()
        self.graph_output = torch.empty_like(self.device_input)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                out = self.model(self.graph_input)
                self.graph_output.copy_(out)
    
    def benchmark_fn(self) -> None:
        """Benchmark: HBM-optimized operations with CUDA graphs."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_hbm", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: HBM memory optimization with CUDA graphs
                # HBM: large contiguous memory access maximizes bandwidth
                # CUDA graphs: replay captured kernels (low overhead)
                # Copy input to static buffer before replay (graph uses static addresses)
                if self.graph_input is None or self.graph_output is None or self.device_input is None:
                    raise RuntimeError("Benchmark not initialized")
                self.graph_input.copy_(self.device_input)
                self.graph.replay()
                self._last_total = float(self.graph_output.sum())
                torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.device_input = None
        self.graph_input = None
        self.graph_output = None
        self.graph = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.device_input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedHbmBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedHbmBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: HBM")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
