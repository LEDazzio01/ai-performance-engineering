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
import torch.nn as nn

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch1.workload_config import WORKLOAD


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineDataParallelismBenchmark(Benchmark):
    """Baseline: Sequential inference without data parallelism (single GPU)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.requests = None
        self.workload = WORKLOAD
        self.num_requests = self.workload.total_requests
        self.hidden_dim = 256
    
    def setup(self) -> None:
        """Setup: Initialize model and requests."""
        torch.manual_seed(42)
        # Baseline: Single GPU inference without data parallelism
        # Data parallelism replicates model across multiple GPUs for higher throughput
        # This baseline processes requests sequentially on one GPU
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(self.device).eval()
        
        # Generate multiple inference requests that will be replayed each iteration
        self.requests = torch.randn(self.num_requests, self.hidden_dim, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential inference without data parallelism."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_data_parallelism", enable=enable_nvtx):
            # Baseline: Sequential processing on single GPU
            # No data parallelism - requests processed one at a time
            # Data parallelism would replicate model across GPUs for parallel processing
            with torch.no_grad():
                for idx in range(self.num_requests):
                    _ = self.model(self.requests[idx : idx + 1])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
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
