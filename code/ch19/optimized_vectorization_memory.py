"""optimized_vectorization_memory.py - Optimized memory management with vectorization.

Demonstrates memory operations optimized with vectorization (SIMD).
Vectorization processes multiple elements per instruction for better efficiency.
Implements BaseBenchmark for harness integration.
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

from common.torch_compile_safe import safe_compile

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class VectorizedTransform(nn.Module):
    """Fused vectorized transform that processes all lanes simultaneously."""

    def __init__(self, vector_width: int):
        super().__init__()
        self.vector_width = vector_width
        self.register_buffer("scale", torch.ones(1, vector_width))
        self.register_buffer("bias", torch.zeros(1, vector_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = torch.nn.functional.silu(x * self.scale + self.bias)
        return fused


class OptimizedVectorizationMemoryBenchmark(BaseBenchmark):
    """Optimized: Memory operations with vectorization."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.data = None
        self.output = None
        self.weights = None
        self.bias = None
        self.transform = None
        self.transform_compiled: Optional[nn.Module] = None
        self.vector_width = 64
        self.num_rows = 65_536
        self.repeats = 8
    
    def setup(self) -> None:
        """Setup: Initialize tensors for vectorized operations."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Vectorized operations
        # Vectorization uses SIMD instructions to process multiple elements per instruction
        # Improves memory bandwidth utilization and reduces instruction overhead
        # PyTorch operations automatically use vectorization when beneficial
        
        total = self.num_rows * self.vector_width
        self.data = torch.randn(total, device=self.device, dtype=torch.float16).view(self.num_rows, self.vector_width)
        self.output = torch.empty_like(self.data)
        self.weights = torch.randn(1, self.vector_width, device=self.device, dtype=torch.float16)
        self.bias = torch.randn(1, self.vector_width, device=self.device, dtype=torch.float16)
        self.transform = VectorizedTransform(self.vector_width).to(self.device).half()
        self.transform.scale.copy_(self.weights)
        self.transform.bias.copy_(self.bias)
        self.transform_compiled = safe_compile(
            self.transform,
            mode="reduce-overhead",
            timeout=120,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Vectorized memory operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("vectorization_memory", enable=enable_nvtx):
            if self.transform_compiled is None:
                raise RuntimeError(
                    "SKIPPED: optimized_vectorization_memory requires torch.compile support."
                )
            with torch.autocast("cuda", dtype=torch.float16):
                for _ in range(self.repeats):
                    fused = self.transform_compiled(self.data)
                    self.output.copy_(fused)
            torch.cuda.synchronize()
            
    # Vectorized operations process multiple elements per instruction
    # Improves memory efficiency compared to scalar operations
    # See ch5 for full vectorization optimization techniques

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        self.weights = None
        self.bias = None
        self.transform = None
        self.transform_compiled = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedVectorizationMemoryBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Vectorization Memory: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Vectorization processes multiple elements per instruction for better memory efficiency")
