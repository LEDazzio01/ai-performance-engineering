"""baseline_continuous_batching.py - Baseline static batching in kernel launches/CUDA graphs context.

Demonstrates static batching with many kernel launches (no CUDA graphs).
Continuous batching: This baseline does not implement continuous batching.
Multiple kernel launches without CUDA graphs optimization.
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

from typing import Optional, List

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class BaselineContinuousBatchingBenchmark(Benchmark):
    """Baseline: Static batching with many kernel launches (no CUDA graphs).
    
    Continuous batching: This baseline does not implement continuous batching.
    Multiple kernel launches without CUDA graphs optimization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.batches = None
    
    def setup(self) -> None:
        """Setup: Initialize model and static batches."""
        torch.manual_seed(42)
        # Baseline: Static batching with many kernel launches
        # Continuous batching: not implemented
        # No CUDA graphs: each batch launches separate kernels
        
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Static batches: fixed sizes, processed sequentially
        # Continuous batching: cannot add/remove requests dynamically
        batch_size = 4
        num_batches = 8
        self.batches = [
            torch.randn(batch_size, 256, device=self.device)
            for _ in range(num_batches)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Static batching with many kernel launches."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Static batching with many kernel launches
                # Continuous batching: not implemented
                # No CUDA graphs: each batch is separate kernel launches
                for batch in self.batches:
                    # Multiple kernel launches per batch (no CUDA graphs)
                    output = self.model(batch)  # Kernel launches for each layer
                    _ = output.sum()  # Additional kernel launch
                    torch.cuda.synchronize()  # Wait before next batch
                
                # Baseline: No continuous batching or CUDA graphs
                # - Static batches (inefficient)
                # - Many kernel launches (high overhead)

    
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
        config=BenchmarkConfig(iterations=10, warmup=2)
    )
    benchmark = BaselineContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Continuous Batching")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

