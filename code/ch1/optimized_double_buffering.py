"""optimized_double_buffering.py - Overlapped memory transfer and computation (optimized).

Demonstrates double buffering where memory transfer and computation overlap.
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


class OptimizedDoubleBufferingBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol.
    
    Demonstrates double buffering: overlap memory transfers with computation.
    Uses two buffers and async streams to hide transfer latency.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.buffer_size = None
        self.host_buffer_a = None
        self.host_buffer_b = None
        self.device_buffer_a = None
        self.device_buffer_b = None
        self.result_buffer_a = None
        self.result_buffer_b = None
        self.h2d_events = None
        self.compute_events = None
        self.N = 10_000_000  # Total elements processed per call (match baseline)
        self.stream_compute = None
        self.stream_h2d = None
        self.stream_d2h = None
        self.num_batches = 2  # Two-stage pipeline to match baseline work
    
    def setup(self) -> None:
        """Setup: Initialize buffers and streams (EXCLUDED from timing)."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        self.buffer_size = self.N // 2
        # Create pinned host memory for efficient async transfers
        self.host_buffer_a = torch.randn(self.buffer_size, pin_memory=True)
        self.host_buffer_b = torch.randn(self.buffer_size, pin_memory=True)
        # Preallocate device buffers
        self.device_buffer_a = torch.empty(self.buffer_size, device=self.device, dtype=torch.float32)
        self.device_buffer_b = torch.empty(self.buffer_size, device=self.device, dtype=torch.float32)
        self.result_buffer_a = torch.empty(self.buffer_size, pin_memory=True)
        self.result_buffer_b = torch.empty(self.buffer_size, pin_memory=True)
        # Create separate streams for transfer and compute
        self.stream_compute = torch.cuda.Stream()
        self.stream_h2d = torch.cuda.Stream()
        self.stream_d2h = torch.cuda.Stream()
        self.h2d_events = [torch.cuda.Event(blocking=False) for _ in range(2)]
        self.compute_events = [torch.cuda.Event(blocking=False) for _ in range(2)]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Double buffering with overlapped transfers and computation.
        
        This pattern achieves better performance by:
        1. Transferring buffer A while computing on buffer B (overlap)
        2. Transferring buffer B while computing on buffer A (overlap)
        Memory transfers and computation happen concurrently.
        """
        # Use conditional NVTX ranges - only enabled when profiling
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range("optimized_double_buffering_overlapped", enable=enable_nvtx):
            buffers = [
                {
                    "host": self.host_buffer_a,
                    "device": self.device_buffer_a,
                    "result": self.result_buffer_a,
                    "h2d_event": self.h2d_events[0],
                    "compute_event": self.compute_events[0],
                },
                {
                    "host": self.host_buffer_b,
                    "device": self.device_buffer_b,
                    "result": self.result_buffer_b,
                    "h2d_event": self.h2d_events[1],
                    "compute_event": self.compute_events[1],
                },
            ]
            
            # Preload first buffer so compute stream has work immediately
            with torch.cuda.stream(self.stream_h2d):
                with nvtx_range("H2D_transfer_async", enable=enable_nvtx):
                    buffers[0]["device"].copy_(buffers[0]["host"], non_blocking=True)
                buffers[0]["h2d_event"].record(self.stream_h2d)

            for batch_idx in range(self.num_batches):
                current_idx = batch_idx % 2
                next_idx = (batch_idx + 1) % 2
                current = buffers[current_idx]
                next_buffer = buffers[next_idx]

                # Ensure compute stream waits for the transfer to complete without global sync
                self.stream_compute.wait_event(current["h2d_event"])
                with torch.cuda.stream(self.stream_compute):
                    with nvtx_range("computation_overlapped", enable=enable_nvtx):
                        # Optimization: Fused operations for better performance
                        current["device"].mul_(2.0).add_(1.0)
                    current["compute_event"].record(self.stream_compute)

                # Kick off async D2H copy once compute is done
                with torch.cuda.stream(self.stream_d2h):
                    self.stream_d2h.wait_event(current["compute_event"])
                    with nvtx_range("D2H_transfer_async", enable=enable_nvtx):
                        current["result"].copy_(current["device"], non_blocking=True)

                # Launch the next H2D transfer to overlap with current compute/D2H
                if batch_idx + 1 < self.num_batches:
                    with torch.cuda.stream(self.stream_h2d):
                        with nvtx_range("H2D_transfer_async", enable=enable_nvtx):
                            next_buffer["device"].copy_(next_buffer["host"], non_blocking=True)
                        next_buffer["h2d_event"].record(self.stream_h2d)

            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_buffer_a = None
        self.host_buffer_b = None
        self.device_buffer_a = None
        self.device_buffer_b = None
        self.result_buffer_a = None
        self.result_buffer_b = None
        if self.stream_compute is not None:
            del self.stream_compute
        if self.stream_h2d is not None:
            del self.stream_h2d
        if self.stream_d2h is not None:
            del self.stream_d2h
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
        if self.result_buffer_a is None:
            return "Result buffer A is None"
        if self.result_buffer_b is None:
            return "Result buffer B is None"
        if self.result_buffer_a.shape[0] != self.buffer_size:
            return f"Result buffer A shape mismatch: expected {self.buffer_size}, got {self.result_buffer_a.shape[0]}"
        if self.result_buffer_b.shape[0] != self.buffer_size:
            return f"Result buffer B shape mismatch: expected {self.buffer_size}, got {self.result_buffer_b.shape[0]}"
        # Check that result values are reasonable
        if not torch.isfinite(self.result_buffer_a).all():
            return "Result buffer A contains non-finite values"
        if not torch.isfinite(self.result_buffer_b).all():
            return "Result buffer B contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedDoubleBufferingBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Double Buffering Benchmark Results:")
    print(f"  Mean time: {result.mean_ms:.3f} ms")
    print(f"  Std dev: {result.std_ms:.3f} ms")
    print(f"  Min time: {result.min_ms:.3f} ms")
    print(f"  Max time: {result.max_ms:.3f} ms")
