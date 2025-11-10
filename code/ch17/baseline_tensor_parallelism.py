"""baseline_tensor_parallelism.py - Baseline without tensor parallelism.

Demonstrates model execution without tensor parallelism.
Tensor parallelism: This baseline processes entire tensors on a single GPU.
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
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class BaselineTensorParallelismBenchmark(Benchmark):
    """Baseline: Model execution without tensor parallelism (single GPU)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input_data = None
        self.batch_size = 256
        self.hidden_size = 2048
    
    def setup(self) -> None:
        """Setup: Initialize model on single GPU."""
        torch.manual_seed(42)
        # Baseline: Model on single GPU without tensor parallelism
        # Tensor parallelism splits tensors across multiple GPUs
        # This baseline processes entire tensors on one GPU
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        ).to(self.device, dtype=torch.float32).eval()
        
        self.input_data = torch.randn(self.batch_size, self.hidden_size, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Model execution without tensor parallelism."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Baseline: Process entire model on single GPU
        # No tensor parallelism - all operations on one device
        with nvtx_range("baseline_tensor_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                output = self.model(self.input_data)
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input_data = None
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
    return BaselineTensorParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
