"""baseline_nvlink.py - Baseline without NVLink optimization. Demonstrates memory transfer without NVLink (uses PCIe). Implements Benchmark protocol for harness integration. """

from __future__ import annotations

import sys
from pathlib import Path

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
)
from ch1.workload_config import WORKLOAD


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineNvlinkBenchmark(Benchmark):
    """Baseline: Memory transfer without NVLink (PCIe only)."""

    def __init__(self):
        self.device = resolve_device()
        self.host_chunks = None
        self.device_buffer = None
        self.N = 16_000_000
        self.num_chunks = max(16, WORKLOAD.prefill_chunks * 2)

    def setup(self) -> None:
        """Setup: Initialize host and device memory."""
        torch.manual_seed(42)
        chunk = self.N // self.num_chunks
        self.host_chunks = [
            torch.randn(chunk, dtype=torch.float32) for _ in range(self.num_chunks)
        ]
        self.device_buffer = torch.empty(chunk, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Memory transfer without NVLink optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nvlink", enable=enable_nvtx):
            # Baseline: Allocate-immediately + synchronous copies (no NVLink/pinned memory).
            total = 0.0
            for host_tensor in self.host_chunks:
                device_tensor = host_tensor.to(self.device, non_blocking=False)
                total += device_tensor.sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._checksum = total


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_chunks = None
        self.device_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.host_chunks is None:
            return "Host tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineNvlinkBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline NVLink (PCIe): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Uses PCIe transfer, not NVLink-optimized")
