"""optimized_streams.py - Optimized CUDA streams for parallel execution in GEMM context.

Demonstrates CUDA streams for parallel execution of independent operations.
Streams: Uses CUDA streams to overlap computation and memory transfers.
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
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")

class OptimizedStreamsBenchmark(Benchmark):
    """Optimized: CUDA streams for parallel execution.
    
    Streams: Uses CUDA streams to overlap computation and memory transfers.
    Improves GPU utilization through parallel execution.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.input1 = None
        self.input2 = None
        self.stream1 = None
        self.stream2 = None
    
    def setup(self) -> None:
        """Setup: Initialize model and CUDA streams."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: CUDA streams for parallel execution
        # Streams allow independent operations to execute concurrently
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.input1 = torch.randn(32, 1024, device=self.device)
        self.input2 = torch.randn(32, 1024, device=self.device)
        
        # Create CUDA streams for parallel execution
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Parallel execution with CUDA streams."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_streams", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: CUDA streams for parallel execution
                # Independent operations execute concurrently on different streams
                # Streams: parallel execution improves GPU utilization
                
                with torch.cuda.stream(self.stream1):
                    output1 = self.model(self.input1)
                
                with torch.cuda.stream(self.stream2):
                    output2 = self.model(self.input2)
                
                # Synchronize streams (streams: wait for completion)
                self.stream1.synchronize()
                self.stream2.synchronize()
                
                # Optimization: CUDA streams benefits
                # - Parallel execution of independent operations
                # - Overlaps computation and memory transfers
                # - Better GPU utilization
                # - Improved throughput through parallelism
                _ = output1 + output2

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input1 = None
        self.input2 = None
        self.stream1 = None
        self.stream2 = None
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
        if self.input1 is None or self.input2 is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedStreamsBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedStreamsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Streams")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
