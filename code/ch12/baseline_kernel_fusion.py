"""baseline_kernel_fusion.py - Separate kernel launches (baseline).

Demonstrates multiple kernel launches with intermediate memory traffic.
Uses PyTorch CUDA extension for accurate GPU timing with CUDA Events.
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
    BenchmarkHarness,
    BenchmarkMode,
)

# Import CUDA extension
from ch12.cuda_extensions import load_kernel_fusion_extension


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class BaselineKernelFusionBenchmark(Benchmark):
    """Separate kernel launches - causes multiple memory round trips (uses CUDA extension)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 1_000_000
        self.iterations = 5
        self._extension = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        self._extension = load_kernel_fusion_extension()
        
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
        # Dry run to pay compilation/initialization cost up front
        self._extension.separate_kernels(self.data, 1)
        torch.cuda.synchronize()
        # Reset data so benchmark iterations always start from same values
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Separate kernel launches (3 memory round trips)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("kernel_fusion", enable=enable_nvtx):
            # Call CUDA extension with separate kernels
            self._extension.separate_kernels(self.data, self.iterations)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,  # Fewer iterations since kernels run internally
            warmup=1,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take 60-90 seconds
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineKernelFusionBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Kernel Fusion (CUDA Extension): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
