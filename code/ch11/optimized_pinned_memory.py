"""optimized_pinned_memory.py - Pinned memory transfer (optimized).

Demonstrates faster CPU-GPU memory transfer using pinned memory.
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


class OptimizedPinnedMemoryBenchmark(Benchmark):
    """Pinned memory transfer - faster CPU-GPU transfers.
    
    Pinned memory (page-locked memory) allows direct memory access (DMA)
    transfers between CPU and GPU, eliminating the need for CPU staging buffers.
    This significantly speeds up H2D and D2H transfers.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.cpu_batches: Optional[list[torch.Tensor]] = None
        self.gpu_batches: Optional[list[torch.Tensor]] = None
        self.transfer_stream = None
        self.total_elements = 20_000_000
        self.num_transfers = 4
        self.chunk_size = self.total_elements // self.num_transfers
    
    def setup(self) -> None:
        """Setup: Initialize CPU tensor with pinned memory."""
        torch.manual_seed(42)
        
        # Optimized: Pinned memory - page-locked CPU memory
        self.cpu_batches = [
            torch.randn(self.chunk_size, dtype=torch.float32, pin_memory=True)
            for _ in range(self.num_transfers)
        ]
        self.gpu_batches = [
            torch.empty(self.chunk_size, device=self.device, dtype=torch.float32)
            for _ in range(self.num_transfers)
        ]
        self.transfer_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Pinned memory transfer."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_pinned_memory_pinned", enable=enable_nvtx):
            # Optimized: Transfer pinned memory on a dedicated stream using
            # non-blocking copies, allowing the default stream to execute work.
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self.transfer_stream):
                for cpu_batch, gpu_batch in zip(self.cpu_batches, self.gpu_batches):
                    gpu_batch.copy_(cpu_batch, non_blocking=True)
            current_stream.wait_stream(self.transfer_stream)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.cpu_batches = None
        self.gpu_batches = None
        self.transfer_stream = None
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
    return OptimizedPinnedMemoryBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Pinned Memory (Pinned): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
