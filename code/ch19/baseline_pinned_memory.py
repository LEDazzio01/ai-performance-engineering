"""baseline_pinned_memory.py - Unpinned memory transfer (baseline).

Demonstrates standard CPU-GPU memory transfer without pinned memory.
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
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class BaselinePinnedMemoryBenchmark(Benchmark):
    """Unpinned memory transfer - slower CPU-GPU transfers."""
    
    def __init__(self):
        self.device = resolve_device()
        self.cpu_data = None
        self.gpu_data = None
        self.device_buffer = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize CPU tensor without pinning."""
        torch.manual_seed(42)
        
        # Baseline: Unpinned memory - standard CPU allocation
        # This requires CPU staging buffer for H2D transfers
        self.cpu_data = torch.randn(self.N, dtype=torch.float32)
        self.device_buffer = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.gpu_data = None
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Unpinned memory transfer."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("pinned_memory", enable=enable_nvtx):
            # Baseline: Transfer unpinned memory to GPU
            # This is slower because it requires CPU staging buffer
            staging = self.cpu_data.clone()
            self.gpu_data = staging.to(self.device, non_blocking=False)
            self.device_buffer.copy_(self.gpu_data)
            self.device_buffer.mul_(1.0001).add_(0.0001)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.cpu_data = None
        self.gpu_data = None
        self.device_buffer = None
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
        if self.gpu_data is None:
            return "GPU tensor not initialized"
        if self.cpu_data is None:
            return "CPU tensor not initialized"
        if self.gpu_data.shape[0] != self.N:
            return f"GPU tensor shape mismatch: expected {self.N}, got {self.gpu_data.shape[0]}"
        if not torch.isfinite(self.gpu_data).all():
            return "GPU tensor contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselinePinnedMemoryBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Pinned Memory (Unpinned): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
