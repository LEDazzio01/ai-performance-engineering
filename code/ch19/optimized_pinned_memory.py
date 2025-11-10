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
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class OptimizedPinnedMemoryBenchmark(Benchmark):
    """Pinned memory transfer - faster CPU-GPU transfers.
    
    Pinned memory (page-locked memory) allows direct memory access (DMA)
    transfers between CPU and GPU, eliminating the need for CPU staging buffers.
    This significantly speeds up H2D and D2H transfers.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.cpu_buffers = None
        self.device_buffers = None
        self.copy_streams = None
        self.compute_stream = None
        self.copy_events = None
        self.gpu_data = None
        self.active_slot = 0
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize CPU tensor with pinned memory."""
        torch.manual_seed(42)
        
        # Optimized: double-buffer pinned memory with overlapping copy/compute streams
        self.cpu_buffers = [
            torch.randn(self.N, dtype=torch.float32, pin_memory=True)
            for _ in range(2)
        ]
        self.device_buffers = [
            torch.empty(self.N, device=self.device, dtype=torch.float32)
            for _ in range(2)
        ]
        self.copy_streams = [torch.cuda.Stream() for _ in range(2)]
        self.compute_stream = torch.cuda.Stream()
        self.copy_events = [torch.cuda.Event(blocking=False) for _ in range(2)]
        self.gpu_data = None
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Pinned memory transfer."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_pinned_memory_pinned", enable=enable_nvtx):
            slot = self.active_slot
            host = self.cpu_buffers[slot]
            device_buf = self.device_buffers[slot]
            copy_stream = self.copy_streams[slot]
            event = self.copy_events[slot]

            # Start async H2D copy on dedicated stream
            with torch.cuda.stream(copy_stream):
                device_buf.copy_(host, non_blocking=True)
                event.record(copy_stream)

            # Overlap compute on previous buffer while copy runs
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(event)
                device_buf.mul_(1.0001).add_(0.0001)

            self.gpu_data = device_buf
            self.active_slot = 1 - slot
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.cpu_buffers = None
        self.device_buffers = None
        self.copy_streams = None
        self.compute_stream = None
        self.copy_events = None
        self.gpu_data = None
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
        if not self.cpu_buffers:
            return "CPU tensor not initialized"
        if self.gpu_data.shape[0] != self.N:
            return f"GPU tensor shape mismatch: expected {self.N}, got {self.gpu_data.shape[0]}"
        if not torch.isfinite(self.gpu_data).all():
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
