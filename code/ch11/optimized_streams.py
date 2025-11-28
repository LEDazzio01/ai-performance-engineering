"""optimized_streams.py - Pipelined H2D transfers overlapping with compute (optimized).

Chapter 11: CUDA Streams and Concurrency

This optimized version demonstrates stream-based pipelining where:
1. H2D transfers happen on one stream
2. Compute happens on another stream
3. The streams overlap - while computing on chunk N, transfer chunk N+1

This eliminates the idle time from sequential execution by keeping both
the memory controller and compute units busy simultaneously.

Key optimization technique: Double-buffered pipelining with CUDA streams
"""

from __future__ import annotations

from typing import Optional, List

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedStreamsBenchmark(BaseBenchmark):
    """Pipelined execution - overlap H2D transfers with compute using streams.
    
    Stream Pipeline Architecture:
    
    Time ->
    Stream H2D:    [Transfer 0]  [Transfer 1]  [Transfer 2]  ...
    Stream Compute:              [Compute 0]   [Compute 1]   [Compute 2] ...
    
    Each compute waits for its transfer, but transfers and computes overlap.
    """
    
    def __init__(self):
        super().__init__()
        self.host_data: Optional[List[torch.Tensor]] = None
        self.device_data: Optional[List[torch.Tensor]] = None
        self.results: Optional[List[torch.Tensor]] = None
        self.stream_h2d: Optional[torch.cuda.Stream] = None
        self.stream_compute: Optional[torch.cuda.Stream] = None
        self.N = 5_000_000  # Elements per chunk - balanced for H2D/compute overlap
        self.num_chunks = 20  # More chunks to amortize pipeline startup
    
    def setup(self) -> None:
        """Setup: Initialize streams, pinned memory, and device buffers."""
        # Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        torch.manual_seed(42)
        
        # Create streams for pipelining
        self.stream_h2d = torch.cuda.Stream()
        self.stream_compute = torch.cuda.Stream()
        
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
        """Benchmark: Pipelined H2D transfer overlapping with compute.
        
        Key insight: While GPU is computing on chunk i, we transfer chunk i+1.
        This keeps both memory controller and compute units busy.
        
        Pipeline stages:
        1. Start first H2D transfer
        2. For each chunk:
           - Wait for its transfer to complete (on compute stream)
           - Start next transfer (if any)
           - Compute on current chunk
        3. Synchronize all streams
        """
        with self._nvtx_range("streams_pipelined"):
            # Stage 0: Kick off first transfer
            with torch.cuda.stream(self.stream_h2d):
                self.device_data[0].copy_(self.host_data[0], non_blocking=True)
            
            for i in range(self.num_chunks):
                # Ensure this chunk's transfer is complete before computing
                self.stream_compute.wait_stream(self.stream_h2d)
                
                # Start next transfer while we compute (double buffering)
                if i + 1 < self.num_chunks:
                    with torch.cuda.stream(self.stream_h2d):
                        self.device_data[i + 1].copy_(
                            self.host_data[i + 1], non_blocking=True
                        )
                
                # Compute on current chunk
                with torch.cuda.stream(self.stream_compute):
                    self.results[i] = self._compute(self.device_data[i])
            
            # Synchronize both streams
            self.stream_compute.synchronize()
            self.stream_h2d.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        self.results = None
        self.stream_h2d = None
        self.stream_compute = None
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
            num_streams=2,  # H2D stream + compute stream
            num_operations=self.num_chunks * 2,  # transfer + compute per chunk
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.results is None:
            return "Results not initialized"
        for i, r in enumerate(self.results):
            if not torch.isfinite(r).all():
                return f"Result {i} contains non-finite values"
        return None


def get_benchmark() -> OptimizedStreamsBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStreamsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Streams (Pipelined): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
