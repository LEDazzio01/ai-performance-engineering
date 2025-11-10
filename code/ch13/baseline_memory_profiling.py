"""baseline_memory_profiling.py - Baseline memory profiling (baseline).

Memory profiling without optimization - tracks memory usage but doesn't optimize.
Identifies memory leaks and peak usage patterns.

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
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class SimpleModel(nn.Module):
    """Simple model for memory profiling."""
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineMemoryProfilingBenchmark(Benchmark):
    """Baseline memory profiling - tracks usage without optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.targets = None
        self.criterion = None
        self.peak_memory_mb = 0.0
        self.batch_size = 32
        self.hidden_dim = 2048
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.cuda.reset_peak_memory_stats()
        
        # Baseline model without memory optimization
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.criterion = nn.MSELoss()
        
        # Warmup
        _ = self.model(self.inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - memory profiling baseline."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_memory_profiling", enable=enable_nvtx):
            # Forward pass (builds computation graph)
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            
            # Backward pass (keeps all intermediate activations)
            loss.backward()
            
            # Track peak memory
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.criterion
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
        if self.peak_memory_mb <= 0:
            return "Peak memory not recorded"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryProfilingBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Memory Profiling: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Peak Memory: {benchmark.peak_memory_mb:.2f} MB")
from common.python.compile_utils import enable_tf32
