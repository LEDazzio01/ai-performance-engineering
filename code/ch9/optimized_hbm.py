"""optimized_hbm.py - Optimized HBM memory access in kernel efficiency/arithmetic intensity context.

Demonstrates HBM memory optimization for high bandwidth utilization.
HBM: Optimizes memory access patterns for HBM high bandwidth.
Maximizes HBM memory bandwidth utilization.
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
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedHbmBenchmark(Benchmark):
    """Optimized: HBM memory optimization for high bandwidth utilization.
    
    HBM: Optimizes memory access patterns for HBM high bandwidth.
    Maximizes HBM memory bandwidth utilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with HBM optimization."""
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)

        self.model = nn.Sequential(
            nn.Linear(1024, 2048, bias=False),
            nn.GELU(),
            nn.Linear(2048, 1024, bias=False),
        ).to(self.device).eval()

        self.batch = 512
        host_input = torch.randn(self.batch, 1024)
        self.host_pinned = host_input.pin_memory()
        self.device_input = host_input.to(self.device, non_blocking=True)
        self.prefetch_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: HBM-optimized operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_hbm", enable=enable_nvtx):
            with torch.no_grad():
                with torch.cuda.stream(self.prefetch_stream):
                    self.device_input.copy_(self.host_pinned, non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)

                fused = self.model(self.device_input)
                if fused.shape[0] == self.device_input.shape[0]:
                    fused = self.model(fused)
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
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
        if self.input is None:
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
