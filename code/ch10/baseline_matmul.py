"""baseline_matmul.py - Baseline without Matmul optimization. Demonstrates memory transfer without Matmul (uses PCIe). Implements Benchmark protocol for harness integration. """

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch10.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineMatmulBenchmark(Benchmark):
    """Baseline: Memory transfer without Matmul (PCIe only)."""

    def __init__(self):
        self.device = resolve_device()
        self.host_data = None
        self.device_data = None
        self.N = 10_000_000

    def setup(self) -> None:
        """Setup: Initialize host and device memory."""
        torch.manual_seed(42)
        # Baseline: Standard host-device transfer without Matmul
        self.host_data = torch.randn(self.N, dtype=torch.float32)
        self.device_data = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Memory transfer without Matmul optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_matmul", enable=enable_nvtx):
            # Baseline: Standard PCIe transfer (no Matmul)
            self.device_data = self.host_data.to(self.device)


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.device_data is None:
            return "Device tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMatmulBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Matmul (PCIe): {result.mean_ms:.3f} ms")
    print(" Note: Uses PCIe transfer, not Matmul-optimized")
