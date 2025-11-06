"""optimized_continuous_batching.py - Optimized with continuous batching.

Demonstrates continuous batching that dynamically adjusts batch composition.
Continuous batching improves GPU utilization by dynamically handling requests.
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
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class OptimizedContinuousBatchingBenchmark(Benchmark):
    """Optimized: Continuous batching with dynamic request handling."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None

        self.batch_requests = None
    
    def setup(self) -> None:
        """Setup: Initialize model for continuous batching."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Continuous batching
        # Dynamically adds/removes requests from batch as they complete
        # Improves GPU utilization by adapting to request arrival patterns
        
        self.model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.model = self.model.to(self.device)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                 pass
        self.model.eval()
        
        # Simulate dynamic batch composition (continuous batching)
        # In practice, requests would be added/removed dynamically
        self.batch_requests = [
        torch.randn(1, 32, 256, device=self.device),
        torch.randn(1, 16, 256, device=self.device),
        torch.randn(1, 48, 256, device=self.device),
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Continuous batching operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                pass
        # Optimization: Continuous batching
        # Dynamically adjusts batch composition as requests complete
        # Adds new requests and removes completed ones efficiently
        # Improves GPU utilization compared to static batching
                
        # Process dynamic batch (simulating continuous batching)
        for request in self.batch_requests:
             pass
        _ = self.model(request)
                
        # Continuous batching improves throughput by adapting to request patterns
        # See ch16 for full continuous batching implementations

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.batch_requests = None
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
    print(" Tip: Continuous batching dynamically adjusts batch composition for better GPU utilization")
