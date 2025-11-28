"""baseline_streams.py - Sequential data transfer and compute (baseline).

Chapter 11: CUDA Streams and Concurrency

This baseline demonstrates sequential execution where each data chunk must:
1. Transfer from host to device (H2D)
2. Compute on the GPU
3. Synchronize before processing the next chunk

The sequential pattern creates bubbles where the GPU compute unit is idle
during memory transfers and vice versa.
"""

from __future__ import annotations

from typing import Optional, List

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineStreamsBenchmark(BaseBenchmark):
    """Sequential execution - no overlap between H2D transfers and compute."""
    
    def __init__(self):
        super().__init__()
        self.host_data: Optional[List[torch.Tensor]] = None
        self.device_data: Optional[List[torch.Tensor]] = None
        self.results: Optional[List[torch.Tensor]] = None
        self.N = 5_000_000  # Elements per chunk - balanced for H2D/compute overlap
        self.num_chunks = 20  # More chunks to amortize pipeline startup
    
    def setup(self) -> None:
        """Setup: Initialize pinned host memory and device buffers."""
        torch.manual_seed(42)
        
        # Create pinned host memory for async transfers
        self.host_data = [
            torch.randn(self.N, dtype=torch.float32).pin_memory() 
            for _ in range(self.num_chunks)
        ]
        
        # Pre-allocate device buffers
        self.device_data = [
            torch.empty(self.N, dtype=torch.float32, device=self.device)
            for _ in range(self.num_chunks)
        ]
        self.results = [
            torch.empty(self.N, dtype=torch.float32, device=self.device)
            for _ in range(self.num_chunks)
        ]
        
        self._synchronize()
        processed = float(self.N * self.num_chunks)
        self.register_workload_metadata(
            tokens_per_iteration=processed,
            requests_per_iteration=float(self.num_chunks),
        )
    
    def _compute(self, data: torch.Tensor) -> torch.Tensor:
        """Compute-intensive operation on GPU data.
        
        Multiple trig operations to ensure compute time is meaningful
        relative to H2D transfer time for proper overlap demonstration.
        """
        result = data
        for _ in range(3):  # Multiple passes to increase compute time
            result = torch.sin(result) * torch.cos(result) + result * 0.1
            result = torch.tanh(result) + torch.sigmoid(result) * 0.5
        return result
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential H2D transfer then compute for each chunk.
        
        Pattern: H2D -> Sync -> Compute -> Sync -> H2D -> Sync -> Compute -> ...
        
        This creates idle time because:
        - GPU compute units are idle during H2D transfers
        - Memory controller is idle during compute
        """
        with self._nvtx_range("baseline_streams_sequential"):
            for i in range(self.num_chunks):
                # Transfer data from host to device (blocking)
                self.device_data[i].copy_(self.host_data[i])
                self._synchronize()  # Wait for transfer to complete
                
                # Compute on device
                self.results[i] = self._compute(self.device_data[i])
                self._synchronize()  # Wait for compute to complete
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        self.results = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        from common.python.benchmark_metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=getattr(self, '_sequential_ms', 10.0),
            overlapped_time_ms=getattr(self, '_overlapped_ms', 5.0),
            num_streams=1,  # Sequential uses default stream only
            num_operations=self.num_chunks * 2,  # transfer + compute per chunk
        )


def get_benchmark() -> BaselineStreamsBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineStreamsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Streams (Sequential): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
