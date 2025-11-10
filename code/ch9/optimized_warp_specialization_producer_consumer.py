"""optimized_warp_specialization.py - Warp specialization with custom Triton kernels.

Demonstrates warp specialization using custom Triton kernels with producer/consumer pattern.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python import triton_compat  # noqa: F401

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Fix library path for CUDA extension (libc10.so)
import os
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(torch_lib_dir):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if torch_lib_dir not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_dir + ':' + current_ld_path

# Try to load CUDA extension for warp specialization
try:
    import warp_specialized_cuda
    CUDA_EXTENSION_AVAILABLE = True
except ImportError as e:
    CUDA_EXTENSION_AVAILABLE = False

# Try to load Triton warp specialization
if TRITON_AVAILABLE:
    try:
        from ch9.warp_specialized_triton import warp_specialized_triton_forward
        TRITON_WARP_SPEC_AVAILABLE = True
    except ImportError:
        TRITON_WARP_SPEC_AVAILABLE = False
else:
    TRITON_WARP_SPEC_AVAILABLE = False

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedWarpSpecializationProducerConsumerBenchmark(Benchmark):
    """
    Optimized: Warp specialization with producer/consumer pattern.
    
    Demonstrates warp specialization for kernel efficiency/arithmetic intensity.
    Producer warps load data, compute warps process, consumer warps store.
    Improves GPU utilization through specialized warp roles.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.producer_model = None
        self.consumer_model = None
        self.input = None
        self.use_triton = TRITON_AVAILABLE
        self.use_cuda_extension = CUDA_EXTENSION_AVAILABLE
    
    def setup(self) -> None:
        """Setup: Initialize models with warp specialization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Use the new TF32 API to avoid legacy/new API mixing under torch.compile
            enable_tf32()
        torch.manual_seed(42)
        
        # Warp specialization: Producer/Consumer pattern
        # REAL warp specialization happens in Triton kernel (warp_specialize=True)
        # Producer stage: Triton kernel replaces this (does ReLU + scaling)
        # Consumer stage: processes the produced data
        # Input is 1024x2048, Triton outputs same shape, so consumer takes 2048 -> 2048
        self.consumer_model = nn.Sequential(
            nn.Linear(2048, 2048),
        ).to(self.device).eval()
        
        # Larger workload to better demonstrate warp specialization benefits
        self.input = torch.randn(1024, 2048, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Warp specialization with custom Triton kernels."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # FAIL FAST: No fallbacks - REAL warp specialization required
        if not TRITON_WARP_SPEC_AVAILABLE:
            raise RuntimeError(
                "REAL warp specialization requires Triton kernels! "
                f"Triton available: {TRITON_WARP_SPEC_AVAILABLE}. "
                "Build Triton kernels for Chapter 9. "
                "NOTE: CUDA extension has Pipeline API synchronization issues (hanging)."
            )
        
        with nvtx_range("optimized_warp_specialization_producer_consumer", enable=enable_nvtx):
            with torch.no_grad():
                # REAL warp specialization: Use Triton kernel with warp_specialize=True
                # Based on Chapter 14's Triton warp specialization examples
                # The Triton kernel replaces the producer stage with warp-specialized processing
                input_flat = self.input.flatten()
                intermediate_flat = warp_specialized_triton_forward(input_flat)
                intermediate = intermediate_flat.view_as(self.input)
                # Apply consumer stage (same shape as input, so consumer processes it)
                output = self.consumer_model(intermediate)
                _ = output.sum()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.consumer_model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            use_subprocess=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.consumer_model is None:
            return "Consumer model not initialized"
        if self.input is None:
            return "Input not initialized"
        if not TRITON_WARP_SPEC_AVAILABLE:
            return "Triton warp specialization kernels not available"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedWarpSpecializationProducerConsumerBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    config = benchmark.get_config()
    config.use_subprocess = False
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Warp Specialization (Producer/Consumer): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
