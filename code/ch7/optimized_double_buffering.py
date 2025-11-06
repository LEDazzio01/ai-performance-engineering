"""optimized_double_buffering.py - Optimized double buffering in memory access/GEMM context.

Demonstrates double buffering for overlapping computation and data transfer.
Double buffering: Uses two buffers to overlap computation and transfer.
Improves GPU utilization through parallel execution.
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
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")

class OptimizedDoubleBufferingBenchmark(Benchmark):
    """Optimized: Double buffering for overlapping computation and transfer.
    
    Double buffering: Uses two buffers to overlap computation and transfer.
    Improves GPU utilization through parallel execution.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.buffer_a = None
        self.buffer_b = None
        self.inputs = None
        self.stream = None
    
    def setup(self) -> None:
        """Setup: Initialize model and double buffers."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Double buffering - two buffers for overlap
        # Double buffering overlaps computation and data transfer
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Double buffering: two buffers
        self.buffer_a = torch.zeros(32, 1024, device=self.device)
        self.buffer_b = torch.zeros(32, 1024, device=self.device)
        self.stream = torch.cuda.Stream()
        
        self.inputs = [
            torch.randn(32, 1024, device=self.device)
            for _ in range(10)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Double buffering with overlap."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_double_buffering", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Double buffering
                # Overlaps computation and data transfer using two buffers
                # Double buffering: ping-pong between buffers
                
                current_buffer = self.buffer_a
                next_buffer = self.buffer_b
                
                for i, inp in enumerate(self.inputs):
                    # Copy next input to next buffer (double buffering: async transfer)
                    with torch.cuda.stream(self.stream):
                        next_buffer.copy_(inp, non_blocking=True)
                    
                    if i > 0:
                        # Process current buffer (double buffering: overlap)
                        _ = self.model(current_buffer)
                    
                    # Swap buffers (double buffering: ping-pong)
                    current_buffer, next_buffer = next_buffer, current_buffer
                    
                    # Synchronize stream (double buffering: ensure transfer complete)
                    self.stream.synchronize()
                
                # Process last buffer
                _ = self.model(current_buffer)
                
                # Optimization: Double buffering benefits
                # - Overlaps computation and transfer
                # - Better GPU utilization
                # - Improved throughput through parallelism

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.buffer_a = None
        self.buffer_b = None
        self.inputs = None
        self.stream = None
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
        if self.buffer_a is None or self.buffer_b is None:
            return "Buffers not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedDoubleBufferingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedDoubleBufferingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Double Buffering")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
