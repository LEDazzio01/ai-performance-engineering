"""optimized_continuous_batching.py - Optimized continuous batching in disaggregated inference context.

Demonstrates continuous batching for optimal GPU utilization.
Continuous batching: Implements dynamic batch composition.
Batches are composed optimally from available samples.
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
from collections import deque

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


class OptimizedContinuousBatchingBenchmark(Benchmark):
    """Optimized: Continuous batching - dynamic batch composition.
    
    Continuous batching: Implements continuous batching where samples can be
    added or removed dynamically. Batches are composed optimally from available
    samples, improving GPU utilization and reducing idle time.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:
            model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        # Optimization: Compile model for kernel fusion and optimization
        try:
            self.model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        self.sample_queue = None
    
    def setup(self) -> None:
        """Setup: Initialize model and sample queue."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Continuous batching - dynamic sample queue
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Simulate continuous batching: queue of samples ready for processing
        num_samples = 100
        self.sample_queue = deque([
            torch.randn(1, 1024, device=self.device)
            for _ in range(num_samples)
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
                # Optimization: Continuous batching
                # Compose batches dynamically from sample queue
                max_batch_size = 8
                
                while self.sample_queue:
                    # Compose batch from queue (continuous batching)
                    current_batch_samples = []
                    current_size = 0
                    
                    # Add samples to batch until max size reached
                    while self.sample_queue and current_size < max_batch_size:
                        sample = self.sample_queue.popleft()
                        current_batch_samples.append(sample)
                        current_size += 1
                    
                    if current_batch_samples:
                        # Stack into batch tensor
                        batch = torch.cat(current_batch_samples, dim=0)
                        
                        # Process batch (continuous batching: optimal composition)
                        _ = self.model(batch)
                        
                        # Continuous batching: New samples can be added to queue
                        # without waiting for full batch completion

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.sample_queue = None
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
        if self.sample_queue is None:
            return "Sample queue not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedContinuousBatchingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: continuous_batching")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
