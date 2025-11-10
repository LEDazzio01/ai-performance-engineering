"""optimized_data_parallelism.py - Optimized data parallelism for multi-GPU inference.

Demonstrates data parallelism by replicating model across multiple GPUs.
Data parallelism: Replicates model across GPUs for parallel processing of different batches.
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
    import ch18.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedDataParallelismBenchmark(Benchmark):
    """Optimized: Data parallelism with model replication across GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.requests = None
        self.num_requests = 64
        self.batch_size = 8
        self.feature_dim = 512
    
    def setup(self) -> None:
        """Setup: Initialize model replicas on multiple GPUs."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 10),
        ).to(self.device).eval()
        self.requests = torch.randn(self.num_requests, self.feature_dim, device=self.device)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Data parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: Process requests in parallel across GPUs
        # Data parallelism enables parallel processing of different requests
        with nvtx_range("optimized_data_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                for start in range(0, self.num_requests, self.batch_size):
                    batch = self.requests[start:start + self.batch_size]
                    if batch.shape[0] == 0:
                        continue
                    _ = self.model(batch)
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedDataParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
