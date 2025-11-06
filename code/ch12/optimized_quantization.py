"""optimized_quantization.py - Optimized quantization with CUDA graphs in kernel launches context.

Demonstrates quantization with CUDA graphs to reduce kernel launch overhead.
Quantization: Uses quantization to reduce numerical precision.
Uses CUDA graphs to capture and replay quantized operations efficiently.
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
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")

class OptimizedQuantizationBenchmark(Benchmark):
    """Optimized: Quantization with CUDA graphs.
    
    Quantization: Uses quantization to reduce numerical precision.
    Uses CUDA graphs to capture and replay quantized operations efficiently.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.input_static = None
        self.graph = None
    
    def setup(self) -> None:
        """Setup: Initialize quantized model with CUDA graphs."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: FP8 Quantization with CUDA graphs
        # FP8 quantization: 2x memory reduction vs FP16, native CUDA support
        # CUDA graphs: capture kernels to reduce launch overhead
        from common.python.quantization_utils import quantize_model_to_fp8, get_quantization_dtype
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        
        # Quantize model to FP8 for CUDA-native quantization
        self.model = quantize_model_to_fp8(self.model, self.device, precision='fp8')
        input_dtype = get_quantization_dtype('fp8')
        self.input = torch.randn(32, 1024, device=self.device, dtype=input_dtype)
        
        # CUDA graphs: warm-up before capture for stable graph creation
        # Warm-up iterations ensure CUDA kernels are compiled and cached
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(self.input)
        torch.cuda.synchronize()
        
        # CUDA graphs: capture quantized operations with static buffer
        # Quantization: capture reduced precision kernels
        # Create static input buffer for graph capture (graph captures tensor addresses)
        self.input_static = self.input.clone()
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                _ = self.model(self.input_static)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Quantized operations with CUDA graphs."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_quantization", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: FP8 Quantization with CUDA graphs
                # FP8 quantization: 2x memory reduction vs FP16, native CUDA support
                # CUDA graphs: replay captured kernels (low overhead)
                # Copy input to static buffer before replay (graph uses static addresses)
                self.input_static.copy_(self.input)
                self.graph.replay()
                
                # Optimization: FP8 Quantization and CUDA graphs benefits
                # - Reduced memory usage (FP8: 2x vs FP16, 4x vs FP32)
                # - Reduced kernel launch overhead (CUDA graphs)
                # - Better performance through graph replay + FP8 computation
                # - Native CUDA support (not CPU-only like qint8)
                _ = self.input.sum()  # Use input for validation

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.input_static = None
        self.graph = None
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

