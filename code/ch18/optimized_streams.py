"""optimized_streams.py - Optimized CUDA streams for parallel execution in FlexAttention/KV cache context.

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
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")

class OptimizedStreamsBenchmark(Benchmark):
    """Optimized: CUDA streams for parallel execution.
    
    Streams: Uses CUDA streams to overlap computation and memory transfers.
    Improves GPU utilization through parallel execution.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input1 = None
        self.input2 = None
        self.output1 = None
        self.output2 = None
        self.stream1 = None
        self.stream2 = None
        self.default_stream = torch.cuda.current_stream(device=self.device)
        self.batch = 8
        self.seq_len = 256
        self.hidden_dim = 512
    
    def setup(self) -> None:
        """Setup: Initialize model and CUDA streams."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: CUDA streams for parallel execution
        # Streams allow independent operations to execute concurrently
        
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True
        ).to(self.device).eval()
        
        self.input1 = torch.randn(self.batch, self.seq_len, self.hidden_dim, device=self.device)
        self.input2 = torch.randn(self.batch, self.seq_len, self.hidden_dim, device=self.device)
        
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
                    self.output1, _ = self.model(self.input1, self.input1, self.input1)
                with torch.cuda.stream(self.stream2):
                    self.output2, _ = self.model(self.input2, self.input2, self.input2)
                
                self.default_stream.wait_stream(self.stream1)
                self.default_stream.wait_stream(self.stream2)
                _ = self.output1 + self.output2

    
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
            iterations=20,
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
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
