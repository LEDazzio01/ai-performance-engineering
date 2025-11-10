"""optimized_cutlass_memory.py - Optimized memory management with CUTLASS.

Demonstrates memory management optimized with CUTLASS GEMM operations.
CUTLASS provides optimized memory access patterns for better efficiency.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch19.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class OptimizedCutlassMemoryBenchmark(Benchmark):
    """Optimized: Memory management with CUTLASS optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.static_a = None
        self.static_b = None
        self.static_out = None
        self.graph = None
        self.capture_stream = None
        self.M, self.N, self.K = 1024, 1024, 1024
    
    def setup(self) -> None:
        """Setup: Initialize matrices and capture a persistent CUTLASS-backed GEMM."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
            torch.backends.cuda.matmul.allow_tf32 = True

        torch.manual_seed(42)

        self.A = torch.randn(self.M, self.K, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.K, self.N, device=self.device, dtype=torch.float16)
        self.C = torch.empty(self.M, self.N, device=self.device, dtype=torch.float16)

        if hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph"):
            try:
                self.graph = torch.cuda.CUDAGraph()
                self.static_a = self.A.clone()
                self.static_b = self.B.clone()
                self.static_out = torch.empty_like(self.C)
                self.capture_stream = torch.cuda.Stream()
                torch.cuda.synchronize()
                with torch.cuda.graph(self.graph, stream=self.capture_stream):
                    self.static_out.copy_(torch.matmul(self.static_a, self.static_b))
            except RuntimeError:
                self.graph = None
                self.static_a = None
                self.static_b = None
                self.static_out = None
                self.capture_stream = None
        else:
            self.graph = None
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUTLASS-optimized memory operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_cutlass_memory", enable=enable_nvtx):
            if self.graph is not None:
                # Refresh dynamic inputs and replay captured CUTLASS GEMM
                self.static_a.copy_(self.A)
                self.static_b.copy_(self.B)
                self.graph.replay()
                self.C.copy_(self.static_out)
            else:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    self.C = torch.matmul(self.A, self.B)
            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.C = None
        self.static_a = None
        self.static_b = None
        self.static_out = None
        self.graph = None
        self.capture_stream = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.C is None:
            return "Output matrix not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCutlassMemoryBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    
    print(f"\nOptimized CUTLASS Memory: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: CUTLASS-optimized memory management improves bandwidth utilization")
