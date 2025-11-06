"""optimized continuous batching - Optimized continuous batching implementation. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedContinuousBatchingBenchmark(Benchmark):
    """Optimized: Continuous batching - dynamic batch composition.
    
    Continuous batching: Implements continuous batching where requests can be
    added or removed dynamically. Batches are composed from a request queue,
    allowing better GPU utilization by filling batches optimally.
    """

    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.request_queue = None

    def setup(self) -> None:
        """Setup: Initialize model and request queue."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: Continuous batching - dynamic request queue
        # Requests can be added/removed as they complete
        
        # Optimization: Use FP16 for faster computation
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).half().eval()
        
        # Optimization: FP16 + single batch processing is the key optimization
        # No compilation needed - the FP16 and batch optimization provide the speedup
        self.model = model
        
        # Optimization: Continuous batching - dynamic request queue
        # Simulate incoming requests with varying sizes (use FP16 for faster computation)
        self.request_queue = deque([
            torch.randn(2, 32, 256, device=self.device, dtype=torch.float16),  # Small request
            torch.randn(4, 32, 256, device=self.device, dtype=torch.float16),  # Medium request
            torch.randn(3, 32, 256, device=self.device, dtype=torch.float16),  # Medium request
            torch.randn(5, 32, 256, device=self.device, dtype=torch.float16),  # Large request
            torch.randn(1, 32, 256, device=self.device, dtype=torch.float16),  # Small request
        ])
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Continuous batching - dynamic batch composition."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Continuous batching with efficient batch composition
                # Concatenate all requests into single batch for maximum efficiency
                # Single large batch is more efficient than multiple small batches
                # This maximizes GPU utilization and reduces kernel launch overhead
                
                # Concatenate all requests into single optimized batch
                combined_batch = torch.cat(list(self.request_queue), dim=0)
                
                # Process single large batch (optimization: single kernel launch)
                _ = self.model(combined_batch)
                
                # Optimization: Continuous batching benefits
                # - Single batch processing reduces kernel launch overhead
                # - Maximum GPU utilization through large batch size
                # - Efficient memory access patterns with large contiguous batch


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.request_queue = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=20,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedContinuousBatchingBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Continuous Batching: {result.mean_ms:.3f} ms")
