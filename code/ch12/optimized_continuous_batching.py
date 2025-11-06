"""optimized_continuous_batching.py - Optimized continuous batching with CUDA graphs in kernel launches context.

Demonstrates continuous batching with CUDA graphs to reduce kernel launch overhead.
Continuous batching: Implements dynamic batch composition with CUDA graphs.
Uses CUDA graphs to capture and replay batches efficiently.
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

from typing import Optional, List, Deque
from collections import deque

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")

class OptimizedContinuousBatchingBenchmark(Benchmark):
    """Optimized: Continuous batching with CUDA graphs.
    
    Continuous batching: Implements dynamic batch composition with CUDA graphs.
    Uses CUDA graphs to capture and replay batches efficiently.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.request_queue = None
        self.graph = None
        self.current_batch = None
        self.batch_static = None
    
    def setup(self) -> None:
        """Setup: Initialize model and continuous batching with CUDA graphs."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Continuous batching with CUDA graphs
        # Continuous batching: dynamic batch composition
        # CUDA graphs: capture kernels to reduce launch overhead
        
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Continuous batching: request queue for dynamic composition
        # Create sample requests
        num_requests = 8
        self.request_queue = deque([
            torch.randn(1, 256, device=self.device)
            for _ in range(num_requests)
        ])
        
        # CUDA graphs: capture batch processing
        # Continuous batching: compose batch from queue
        max_batch_size = 4
        self.current_batch = torch.cat([
            self.request_queue[i] for i in range(min(max_batch_size, len(self.request_queue)))
        ], dim=0)
        
        # CUDA graphs: warm-up before capture for stable graph creation
        # Warm-up iterations ensure CUDA kernels are compiled and cached
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(self.current_batch)
        torch.cuda.synchronize()
        
        # CUDA graphs: capture batch processing with static buffer
        # Create static batch buffer for graph capture (graph captures tensor addresses)
        self.batch_static = self.current_batch.clone()
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                _ = self.model(self.batch_static)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Continuous batching with CUDA graphs."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Continuous batching with CUDA graphs
                # Continuous batching: compose batch from request queue
                # CUDA graphs: replay captured kernels (low overhead)
                
                # Compose batch from queue (continuous batching)
                batch_size = min(4, len(self.request_queue))
                if batch_size > 0:
                    batch = torch.cat([
                        self.request_queue[i] for i in range(batch_size)
                    ], dim=0)
                    
                    # CUDA graphs: replay captured kernels
                    # Continuous batching: dynamic batch processing
                    if batch.shape == self.current_batch.shape:
                        # Reuse graph if batch size matches
                        # Copy to static buffer before replay (graph uses static addresses)
                        self.batch_static.copy_(batch)
                        self.graph.replay()
                    else:
                        # Process directly if size differs (continuous batching flexibility)
                        _ = self.model(batch)
                
                # Optimization: Continuous batching and CUDA graphs benefits
                # - Dynamic batch composition (continuous batching)
                # - Reduced kernel launch overhead (CUDA graphs)
                # - Better GPU utilization

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.request_queue = None
        self.graph = None
        self.current_batch = None
        self.batch_static = None
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
        if self.request_queue is None:
            return "Request queue not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedContinuousBatchingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=2)
    )
    benchmark = OptimizedContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Continuous Batching")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()

