"""baseline_memory_transfer.py - Traditional PCIe memory transfer (baseline).

Demonstrates traditional CPU-GPU memory transfer over PCIe.
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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2")
    return torch.device("cuda")


class BaselineMemoryTransferBenchmark(Benchmark):
    """Traditional PCIe memory transfer - slower path."""
    
    def __init__(self):
        self.device = resolve_device()
        self.host_data = None
        self.device_data = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Host memory (CPU)
        self.host_data = torch.randn(self.N, dtype=torch.float32, pin_memory=True)
        # Device memory (GPU)
        self.device_data = torch.empty(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Traditional H2D transfer over PCIe."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("memory_transfer", enable=enable_nvtx):
            # Traditional synchronous copy (simulates PCIe transfer)
            self.device_data.copy_(self.host_data, non_blocking=False)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.device_data is None:
            return "Device data tensor not initialized"
        if self.host_data is None:
            return "Host data tensor not initialized"
        if self.device_data.shape[0] != self.N:
            return f"Device data size mismatch: expected {self.N}, got {self.device_data.shape[0]}"
        if not torch.isfinite(self.device_data).all():
            return "Device data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryTransferBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Memory Transfer: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")


