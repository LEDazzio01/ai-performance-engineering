"""optimized_cuda_graphs.py - Optimized CUDA graphs for reduced launch overhead.

Demonstrates CUDA graphs for capturing and replaying sequences of operations.
CUDA graphs: Captures and replays operation sequences to reduce launch overhead.
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

from typing import Optional, List

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class OptimizedCudaGraphsBenchmark(Benchmark):
    """Optimized: CUDA graphs for reduced launch overhead."""

    def __init__(self):
        self.device = resolve_device()
        self.pipeline: Optional[nn.Sequential] = None
        self.static_input: Optional[torch.Tensor] = None
        self._graph_callable = None
        self.batch_size = 16
        self.hidden_dim = 256
        self.inputs: Optional[List[torch.Tensor]] = None

    def setup(self) -> None:
        """Setup: Initialize model and capture CUDA graph."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        layers = []
        for _ in range(4):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
        self.pipeline = nn.Sequential(*layers).to(self.device).eval()
        self.static_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)

        if hasattr(torch.cuda, "make_graphed_callables"):
            self._graph_callable = torch.cuda.make_graphed_callables(self.pipeline, (self.static_input,))
        else:
            self._graph_callable = None

        self.inputs = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
            for _ in range(8)
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: CUDA graph replay."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.pipeline is None or self.static_input is None or self.inputs is None:
            raise RuntimeError("CUDA graph benchmark not initialized")

        with nvtx_range("optimized_cuda_graphs", enable=enable_nvtx):
            for inp in self.inputs:
                self.static_input.copy_(inp)
                if self._graph_callable is not None:
                    self._graph_callable(self.static_input)
                else:
                    with torch.no_grad():
                        _ = self.pipeline(self.static_input)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.pipeline = None
        self.inputs = None
        self.static_input = None
        self._graph_callable = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=12,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.pipeline is None:
            return "Model not initialized"
        if self._graph_callable is None:
            return "CUDA graph callable not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCudaGraphsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
