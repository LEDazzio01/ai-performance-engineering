"""baseline_distributed_ilp.py - Baseline single-GPU ILP (no distributed).

Demonstrates ILP operations on single GPU without distributed computing.
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


class BaselineDistributedILPBenchmark(Benchmark):
    """Baseline: Single-GPU ILP (no distributed computing)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.N = self.workload.distributed_elements_for_mode(self.smoke_test)
        self.micro_chunks = self.workload.distributed_chunks_for_mode(self.smoke_test)
        self._scale = 1.1
        self._bias = 0.5
   
    def setup(self) -> None:
        """Setup: Initialize single-GPU tensors."""
        torch.manual_seed(42)
        # Baseline: Single-GPU operation
        # Distributed computing uses multiple GPUs for parallel ILP operations
        # This baseline uses only one GPU
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Single-GPU ILP operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_distributed_ilp", enable=enable_nvtx):
            chunk_size = max(1, self.N // self.micro_chunks)
            cpu_device = torch.device("cpu")
            for chunk_start in range(0, self.N, chunk_size):
                chunk_end = min(self.N, chunk_start + chunk_size)
                chunk = self.input[chunk_start:chunk_end]
                # Baseline: round-trip through host memory before compute.
                host_chunk = chunk.to(cpu_device, non_blocking=False)
                host_result = torch.addcmul(host_chunk, host_chunk, host_chunk, value=self._bias)
                host_result.mul_(self._scale)
                self.output[chunk_start:chunk_end].copy_(host_result.to(self.device))
                torch.cuda.synchronize()
    
    
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
    return BaselineDistributedILPBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Distributed ILP (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Note: Single-GPU operation, no distributed computing")
