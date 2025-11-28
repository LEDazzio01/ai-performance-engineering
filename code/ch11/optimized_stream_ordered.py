"""optimized_stream_ordered.py - Optimized multi-stream overlap example.

Demonstrates launching work across multiple CUDA streams with explicit events."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


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
        self.static_inputs: Optional[list[torch.Tensor]] = None
        self.static_outputs: Optional[list[torch.Tensor]] = None
        self.graphs: Optional[list[torch.cuda.CUDAGraph]] = None
    
    def setup(self) -> None:
        """Setup: initialize lightweight model and per-stream buffers."""
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
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        # Capture per-stream CUDA graphs to reduce launch overhead in the optimized path.
        self.static_inputs = [torch.empty_like(self.device_inputs[0]) for _ in range(self.num_streams)]
        self.static_outputs = [torch.empty_like(self.device_outputs[0]) for _ in range(self.num_streams)]
        self.graphs = []
        # Initialize libraries outside capture to avoid CUBLAS init inside the graph.
        for stream, s_in, s_out in zip(self.streams, self.static_inputs, self.static_outputs):
            with torch.cuda.stream(stream):
                s_in.zero_()
                s_out.copy_(self.model(s_in))
        torch.cuda.synchronize()
        for stream, s_in, s_out in zip(self.streams, self.static_inputs, self.static_outputs):
            g = torch.cuda.CUDAGraph()
            # Use a warm start so the graph captures a stable path.
            s_in.copy_(self.device_inputs[0])
            with torch.cuda.graph(g, stream=stream):
                s_out.copy_(self.model(s_in))
            self.graphs.append(g)
        self._synchronize()
        tokens = float(self.batch_size * self.hidden_dim * self.num_requests)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.num_requests),
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Launch work on dedicated streams to overlap execution."""
        with self._nvtx_range("optimized_stream_ordered"):
            with torch.no_grad():
                assert self.streams is not None
                assert self.host_requests is not None
                assert self.host_outputs is not None
                assert self.device_inputs is not None
                assert self.device_outputs is not None
                assert self.static_inputs is not None
                assert self.static_outputs is not None
                assert self.graphs is not None
                for idx, (h_req, h_out) in enumerate(zip(self.host_requests, self.host_outputs)):
                    slot = idx % self.num_streams
                    stream = self.streams[slot]
                    s_in = self.static_inputs[slot]
                    s_out = self.static_outputs[slot]
                    graph = self.graphs[slot]
                    with torch.cuda.stream(stream):
                        s_in.copy_(h_req, non_blocking=True)
                        graph.replay()
                        h_out.copy_(s_out, non_blocking=True)
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
        self.static_inputs = None
        self.static_outputs = None
        self.graphs = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=80,
            warmup=8,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_stream_metrics
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
