"""baseline_continuous_batching.py - Baseline static batching in inference/profiling context.

Demonstrates static batching where batches are processed sequentially.
Continuous batching: This baseline does not implement continuous batching.
Batches wait for full completion before next batch starts.
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
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class BaselineContinuousBatchingBenchmark(Benchmark):
    """Baseline: Static batching - batches processed sequentially.
    
    Continuous batching: This baseline does not implement continuous batching.
    Batches are processed one at a time, with full synchronization between batches.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.batches = None
    
    def setup(self) -> None:
        """Setup: Initialize model and static batches."""
        torch.manual_seed(42)
        # Baseline: Static batching - fixed batch sizes, processed sequentially
        # Continuous batching allows dynamic batch composition
        # This baseline does not use continuous batching
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Static batches - fixed sizes, processed sequentially
        batch_size = 8
        num_batches = 10
        self.batches = [
            torch.randn(batch_size, 1024, device=self.device)
            for _ in range(num_batches)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Static batching - sequential processing."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Process batches sequentially
                # Each batch must complete fully before next batch starts
                # No continuous batching: cannot add/remove samples mid-batch
                for batch in self.batches:
                    _ = self.model(batch)
                    torch.cuda.synchronize()  # Wait for completion
                
                # Baseline: No continuous batching
                # Fixed batch sizes, sequential processing
                # Poor GPU utilization when batches have varying sizes

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.batches = None
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
        if self.batches is None:
            return "Batches not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineContinuousBatchingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: continuous_batching")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
