"""baseline_double_buffering.py - Baseline without double buffering in kernel efficiency/arithmetic intensity context.

Demonstrates operations without double buffering optimization.
Double buffering: This baseline does not use double buffering.
Sequential processing without overlapping computation and data transfer.
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


class BaselineDoubleBufferingBenchmark(Benchmark):
    """Baseline: Sequential processing without double buffering.
    
    Double buffering: This baseline does not use double buffering.
    Sequential processing without overlapping computation and data transfer.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.buffer = None
        self.inputs = None
    
    def setup(self) -> None:
        """Setup: Initialize model and single buffer."""
        torch.manual_seed(42)
        # Baseline: Single buffer - sequential processing
        # Double buffering uses two buffers to overlap computation and transfer
        # This baseline does not use double buffering
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Single buffer (no double buffering)
        self.buffer = torch.zeros(32, 1024, device=self.device)
        self.inputs = [
            torch.randn(32, 1024, device=self.device)
            for _ in range(10)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential processing without double buffering."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_double_buffering", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Sequential processing (no double buffering)
                # Process and transfer sequentially
                # Double buffering would overlap computation and transfer
                for inp in self.inputs:
                    # Copy to buffer (sequential)
                    self.buffer.copy_(inp)
                    torch.cuda.synchronize()  # Wait for copy
                    
                    # Process (sequential)
                    _ = self.model(self.buffer)
                    torch.cuda.synchronize()  # Wait for computation
                
                # Baseline: No double buffering
                # Sequential processing (inefficient)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.buffer = None
        self.inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.buffer is None:
            return "Buffer not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineDoubleBufferingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=2)
    )
    benchmark = BaselineDoubleBufferingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Double Buffering")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

