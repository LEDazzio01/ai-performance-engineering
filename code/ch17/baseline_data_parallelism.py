"""baseline_data_parallelism.py - Baseline sequential inference without data parallelism.

Demonstrates single-GPU inference without data parallelism.
Data parallelism: This baseline processes requests sequentially on a single GPU.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional, List

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch15.baseline_data_parallelism import _build_mlp


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class BaselineDataParallelismBenchmark(Benchmark):
    """Baseline: Sequential inference without data parallelism (single GPU)."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.requests: Optional[List[torch.Tensor]] = None
        self.num_requests = 128
        self.hidden_dim = 256

    def setup(self) -> None:
        """Setup: Initialize model and requests."""
        torch.manual_seed(42)
        self.model = _build_mlp(self.hidden_dim).to(self.device, dtype=torch.float32).eval()

        self.requests = [
            torch.randn(1, self.hidden_dim, device=self.device, dtype=torch.float32)
            for _ in range(self.num_requests)
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Sequential inference without data parallelism."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.model is None or self.requests is None:
            raise RuntimeError("Model/requests not initialized")

        with nvtx_range("baseline_data_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                for request in self.requests:
                    _ = self.model(request)
                    torch.cuda.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=12,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if not self.requests:
            return "Requests not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineDataParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
