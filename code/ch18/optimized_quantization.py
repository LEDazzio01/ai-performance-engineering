"""optimized_quantization.py - Optimized quantization in FlexAttention/KV cache context.

Demonstrates quantization for reduced precision operations.
Quantization: Uses quantization to reduce numerical precision.
Reduces memory usage and improves performance.
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

class OptimizedQuantizationBenchmark(Benchmark):
    """Optimized: Quantization for reduced precision operations.
    
    Quantization: Uses quantization to reduce numerical precision.
    Reduces memory usage and improves performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize quantized model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Quantization - reduced precision
        # Quantization reduces precision (e.g., INT8, FP8) for performance/memory
        
        # Optimization: FP8 Quantization on CUDA (native support on GB10/H100+)
        # FP8 provides 2x memory reduction vs FP16, 4x vs FP32
        # This is the proper way to do quantization on CUDA, not CPU-only qint8
        from common.python.quantization_utils import quantize_model_to_fp8, get_quantization_dtype
        
        hidden_dim = 256
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Quantize model to FP8 for CUDA-native quantization
        self.model = quantize_model_to_fp8(self.model, self.device, precision='fp8')
        input_dtype = get_quantization_dtype('fp8')
        self.input = torch.randn(4, 128, hidden_dim, device=self.device, dtype=input_dtype)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Quantized operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_quantization", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: FP8 Quantization on CUDA
                # Uses FP8 quantized model (native CUDA support on GB10/H100+)
                # FP8 provides 2x memory reduction vs FP16, 4x vs FP32
                output, _ = self.model(self.input, self.input, self.input)
                
                # Optimization: FP8 Quantization benefits
                # - Reduced memory usage (FP8: 2x vs FP16, 4x vs FP32)
                # - Improved performance (faster FP8 computation on Tensor Cores)
                # - Better memory bandwidth utilization (less data to transfer)
                # - Native CUDA support (not CPU-only like qint8)
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
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
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedQuantizationBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedQuantizationBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Quantization")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
