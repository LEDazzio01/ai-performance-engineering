"""optimized_stream_ordered.py - Optimized multi-stream overlap example.

Demonstrates launching work across multiple CUDA streams with explicit events.
Stream-ordered allocators: Uses dedicated streams to overlap request processing.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")

class OptimizedStreamOrderedBenchmark(Benchmark):
    """Optimized: Overlap work across multiple CUDA streams."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.requests = None
        self.outputs = None
        self.streams = None
        self.num_streams = 8
        self.hidden_dim = 1024
        self.batch_size = 64
    
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
        self.requests = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(self.num_streams)
        ]
        self.outputs = [
            torch.empty_like(req) for req in self.requests
        ]
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Launch work on dedicated streams to overlap execution."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_stream_ordered", enable=enable_nvtx):
            with torch.no_grad():
                for stream, request, output in zip(self.streams, self.requests, self.outputs):
                    with torch.cuda.stream(stream):
                        output.copy_(self.model(request))
                for stream in self.streams:
                    stream.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
        self.outputs = None
        self.streams = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedStreamOrderedBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Stream-Ordered: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Launching work on separate CUDA streams overlaps request processing")
