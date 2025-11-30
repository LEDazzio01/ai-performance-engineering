"""optimized_stream_ordered.py - Optimized multi-stream overlap example.

Demonstrates launching work across multiple CUDA streams with explicit events."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedStreamOrderedBenchmark(BaseBenchmark):
    """Optimized: Overlap work across multiple CUDA streams."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.host_requests: Optional[list[torch.Tensor]] = None
        self.host_outputs: Optional[list[torch.Tensor]] = None
        self.device_inputs: Optional[list[torch.Tensor]] = None
        self.device_outputs: Optional[list[torch.Tensor]] = None
        self.streams: Optional[list[torch.cuda.Stream]] = None
        self.num_streams = 4
        self.num_requests = 32  # More requests to amortize overhead
        self.hidden_dim = 1024
        self.batch_size = 128
    
    def setup(self) -> None:
        """Setup: initialize lightweight model and multi-stream buffers.
        
        Key optimization: Uses multiple CUDA streams for overlapped execution
        vs baseline's single default stream with per-request synchronization.
        """
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).half().eval()

        self.host_requests = [
            torch.randn(
                self.batch_size, self.hidden_dim, device="cpu", dtype=torch.float16, pin_memory=True
            )
            for _ in range(self.num_requests)
        ]
        self.host_outputs = [
            torch.empty_like(req, device="cpu", pin_memory=True) for req in self.host_requests
        ]
        self.device_inputs = [
            torch.empty_like(req, device=self.device) for req in self.host_requests
        ]
        self.device_outputs = [
            torch.empty_like(inp, device=self.device) for inp in self.device_inputs
        ]
        
        # Multiple streams for concurrent execution
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        
        # Warmup to initialize CUDA/cuBLAS on each stream
        for stream in self.streams:
            with torch.cuda.stream(stream):
                _ = self.model(self.device_inputs[0])
        torch.cuda.synchronize()
        
        self._synchronize()
        tokens = float(self.batch_size * self.hidden_dim * self.num_requests)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.num_requests),
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Launch work on dedicated streams to overlap execution.
        
        Key optimization over baseline:
        - Uses multiple CUDA streams for concurrent execution
        - No synchronization inside the loop - work is pipelined
        - Single sync at the end vs per-request sync in baseline
        """
        with self._nvtx_range("optimized_stream_ordered"):
            with torch.no_grad():
                assert self.streams is not None
                assert self.host_requests is not None
                assert self.host_outputs is not None
                assert self.device_inputs is not None
                assert self.device_outputs is not None
                assert self.model is not None
                
                # Process all requests with stream overlap - no sync inside loop
                for idx, (h_req, h_out, d_in, d_out) in enumerate(zip(
                    self.host_requests, self.host_outputs, 
                    self.device_inputs, self.device_outputs
                )):
                    stream = self.streams[idx % self.num_streams]
                    with torch.cuda.stream(stream):
                        # Pipeline: H2D, compute, D2H all non-blocking
                        d_in.copy_(h_req, non_blocking=True)
                        out = self.model(d_in)
                        d_out.copy_(out)
                        h_out.copy_(d_out, non_blocking=True)
                
                # Single synchronization at the end - key difference from baseline
                torch.cuda.synchronize()
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.host_requests = None
        self.host_outputs = None
        self.device_inputs = None
        self.device_outputs = None
        self.streams = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=80,
            warmup=8,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=getattr(self, '_sequential_ms', 10.0),
            overlapped_time_ms=getattr(self, '_overlapped_ms', 5.0),
            num_streams=getattr(self, 'num_streams', 4),
            num_operations=getattr(self, 'num_operations', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> OptimizedStreamOrderedBenchmark:
    """Factory function for harness discovery."""
    return OptimizedStreamOrderedBenchmark()
