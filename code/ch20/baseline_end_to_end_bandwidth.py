"""baseline_end_to_end_bandwidth.py - Baseline end-to-end bandwidth (baseline).

Baseline end-to-end bandwidth analysis without optimizations.
Sequential processing without memory access optimizations.

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
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimplePipeline(nn.Module):
    """Simple inference pipeline."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineEndToEndBandwidthBenchmark(Benchmark):
    """Baseline end-to-end bandwidth - sequential processing."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.outputs = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self.num_batches = 10
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        
        # Baseline: FP32, no optimizations
        self.model = SimplePipeline(hidden_dim=self.hidden_dim).to(self.device).eval()
        
        self.inputs = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
            for _ in range(self.num_batches)
        ]
        self.outputs = []
        
        # Warmup
        for inp in self.inputs[:3]:
            _ = self.model(inp)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - baseline end-to-end."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_end_to_end_bandwidth", enable=enable_nvtx):
            # Sequential processing without optimizations
            self.outputs = []
            for inp in self.inputs:
                out = self.model(inp)
                self.outputs.append(out)
            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.outputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None and len(self.outputs) == self.num_batches


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineEndToEndBandwidthBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline End-to-End Bandwidth: {result.mean_ms:.3f} ms")

