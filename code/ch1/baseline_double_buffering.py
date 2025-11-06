"""baseline_double_buffering.py - Sequential memory transfer and computation (baseline).

Demonstrates sequential execution where memory transfer and computation don't overlap.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineDoubleBufferingBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol.
    
    Demonstrates sequential execution: transfer data, then compute, then transfer back.
    No overlap between memory transfers and computation.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.host_data = None
        self.device_data = None
        self.result = None
        self.N = 10_000_000  # 10M elements
        self.stream = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and stream (EXCLUDED from timing)."""
        torch.manual_seed(42)
        # Create pinned host memory for efficient transfers
        self.host_data = torch.randn(self.N, pin_memory=True)
        # Preallocate device memory
        self.device_data = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.result = torch.empty(self.N, pin_memory=True)
        # Use default stream (synchronous)
        self.stream = torch.cuda.current_stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential memory transfer and computation.
        
        This pattern shows poor performance because:
        1. CPU waits for H2D transfer to complete
        2. GPU computes
        3. CPU waits for D2H transfer to complete
        No overlap between transfers and computation.
        """
        # Use conditional NVTX ranges - only enabled when profiling
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range("baseline_double_buffering_sequential", enable=enable_nvtx):
            # Step 1: Transfer host to device (synchronous)
            with nvtx_range("H2D_transfer", enable=enable_nvtx):
                self.device_data.copy_(self.host_data, non_blocking=False)
                torch.cuda.synchronize()  # Wait for transfer

            # Step 2: Compute on device
            with nvtx_range("computation", enable=enable_nvtx):
                self.device_data = self.device_data * 2.0 + 1.0
                torch.cuda.synchronize()  # Wait for computation

            # Step 3: Transfer device to host (synchronous)
            with nvtx_range("D2H_transfer", enable=enable_nvtx):
                self.result.copy_(self.device_data, non_blocking=False)
                torch.cuda.synchronize()  # Wait for transfer
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        self.result = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.result is None:
            return "Result tensor is None"
        if self.device_data is None:
            return "Device data tensor is None"
        if self.result.shape[0] != self.N:
            return f"Result shape mismatch: expected {self.N}, got {self.result.shape[0]}"
        # Check that result values are reasonable (device_data should be host_data * 2.0 + 1.0)
        if not torch.isfinite(self.result).all():
            return "Result contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineDoubleBufferingBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Double Buffering Benchmark Results:")
    print(f"  Mean time: {result.mean_ms:.3f} ms")
    print(f"  Std dev: {result.std_ms:.3f} ms")
    print(f"  Min time: {result.min_ms:.3f} ms")
    print(f"  Max time: {result.max_ms:.3f} ms")

