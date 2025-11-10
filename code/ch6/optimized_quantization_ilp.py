"""optimized_quantization_ilp.py - Optimized ILP with quantization.

Demonstrates ILP optimization using quantization (lower precision) for higher throughput.
Quantization enables more operations per cycle, improving instruction-level parallelism.
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

class OptimizedQuantizationILPBenchmark(Benchmark):
    """Optimized: Quantized ILP for higher throughput."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.N = self.workload.quantization_elements_for_mode(self.smoke_test)
    
    def setup(self) -> None:
        """Setup: Initialize quantized tensors."""
        
        torch.manual_seed(42)
        # Optimization: Quantized operations for higher ILP throughput
        # Quantization reduces precision (FP16, INT8) to enable more operations per cycle
        # Lower precision enables better ILP by allowing more parallel operations
        # FP16 enables 2x more operations than FP32, improving ILP
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float16)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Quantized ILP operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_quantization_ilp", enable=enable_nvtx):
    # Optimization: Quantized ILP
    # Quantization uses lower precision (FP16, INT8) to improve throughput
    # Lower precision enables more operations per cycle, improving ILP
    # FP16 enables 2x more operations than FP32 per SM cycle
    # This improves instruction-level parallelism through higher throughput
            self.output = self.input * 2.0 + 1.0
    # Quantized operations enable higher ILP throughput
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
    return OptimizedQuantizationILPBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Quantization ILP (FP16): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Tip: Quantization enables higher ILP throughput through lower precision operations")
