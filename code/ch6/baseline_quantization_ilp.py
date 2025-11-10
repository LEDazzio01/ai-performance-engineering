"""baseline_quantization_ilp.py - Baseline ILP without quantization.

Demonstrates ILP operations using full precision (FP32) without quantization.
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


class BaselineQuantizationILPBenchmark(Benchmark):
    """Baseline: Full precision ILP (no quantization)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.N = self.workload.quantization_elements_for_mode(self.smoke_test)
    
    def setup(self) -> None:
        """Setup: Initialize full precision tensors."""
        torch.manual_seed(42)
        # Baseline: Full precision (FP32) operations
        # Quantization reduces precision (FP16, INT8, etc.) to improve ILP throughput
        # This baseline uses full FP32 precision
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Full precision ILP operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_quantization_ilp", enable=enable_nvtx):
            # Baseline: Full precision (FP32) ILP
            # Quantization uses lower precision (FP16, INT8) to improve throughput
            # Lower precision enables more operations per cycle, improving ILP
            # This baseline uses FP32 which limits ILP throughput
            self.output = self.input * 2.0 + 1.0
            # Full precision: Lower ILP throughput due to precision overhead
            # See ch14 for full quantization implementations

    
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
    return BaselineQuantizationILPBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Quantization ILP (FP32): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Note: Full precision operations, no quantization")
