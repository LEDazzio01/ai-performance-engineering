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
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class BaselinePinnedMemoryBenchmark(Benchmark):
    """Unpinned memory transfer - slower CPU-GPU transfers."""
    
    def __init__(self):
        self.device = resolve_device()
        self.cpu_batches: Optional[list[torch.Tensor]] = None
        self.gpu_batches: Optional[list[torch.Tensor]] = None
        self.total_elements = 20_000_000  # ~80 MB per iteration in float32
        self.num_transfers = 4
        self.chunk_size = self.total_elements // self.num_transfers
    
    def setup(self) -> None:
        """Setup: Initialize CPU tensor without pinning."""
        torch.manual_seed(42)
        
        # Baseline: Unpinned memory - standard CPU allocation
        self.cpu_batches = [
            torch.randn(self.chunk_size, dtype=torch.float32)
            for _ in range(self.num_transfers)
        ]
        self.gpu_batches = [
            torch.empty(self.chunk_size, device=self.device, dtype=torch.float32)
            for _ in range(self.num_transfers)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Unpinned memory transfer."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("pinned_memory", enable=enable_nvtx):
            # Baseline: Transfer unpinned memory to GPU using the default stream.
            # Non-blocking copies are not allowed because the tensors are pageable.
            for cpu_batch, gpu_batch in zip(self.cpu_batches, self.gpu_batches):
                gpu_batch.copy_(cpu_batch, non_blocking=False)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.cpu_batches = None
        self.gpu_batches = None
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
        if not self.gpu_batches or not self.cpu_batches:
            return "Batches not initialized"
        for tensor in self.gpu_batches:
            if tensor.shape[0] != self.chunk_size:
                return "GPU tensor shape mismatch"
            if not torch.isfinite(tensor).all():
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
