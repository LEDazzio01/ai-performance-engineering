"""optimized_distributed_ilp.py - Optimized distributed ILP.

Demonstrates distributed ILP across multiple GPUs for parallel processing.
Note: Full distributed requires multi-GPU setup; this demonstrates the concept.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from typing import Optional
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch6.workload_config import WORKLOAD, is_smoke_test

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedDistributedILPBenchmark(Benchmark):
    """Optimized: Distributed ILP across multiple GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.N = self.workload.distributed_elements_for_mode(self.smoke_test)
        self.micro_chunks = self.workload.distributed_chunks_for_mode(self.smoke_test)
        self.streams = [
            torch.cuda.Stream() for _ in range(self.workload.distributed_streams)
        ]
        self._scale = 1.1
        self._bias = 0.5
    
    def setup(self) -> None:
        """Setup: Initialize for distributed ILP."""
        
        torch.manual_seed(42)
        # Optimization: Distributed ILP
        # Uses multiple GPUs for parallel ILP operations
        # Enables larger workloads through distributed parallelism
        # For ch6 single-GPU, we demonstrate the concept
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Distributed ILP operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_distributed_ilp", enable=enable_nvtx):
            chunk_size = max(1, (self.N + len(self.streams) - 1) // len(self.streams))
            ranges = []
            for start in range(0, self.N, chunk_size):
                ranges.append((start, min(self.N, start + chunk_size)))

            for chunk_idx, (chunk_start, chunk_end) in enumerate(ranges):
                stream = self.streams[chunk_idx % len(self.streams)]
                with torch.cuda.stream(stream):
                    chunk = self.input[chunk_start:chunk_end]
                    fused = torch.addcmul(chunk, chunk, chunk, value=self._bias)
                    fused = fused * self._scale + torch.tanh(chunk) * 0.1
                    self.output[chunk_start:chunk_end].copy_(fused)

            torch.cuda.synchronize()
            self.output = self.output.contiguous()
    
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedDistributedILPBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Distributed ILP: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Tip: Distributed ILP enables scaling instruction-level parallelism across multiple GPUs")
