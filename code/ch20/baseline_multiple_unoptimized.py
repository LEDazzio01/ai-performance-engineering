"""baseline_multiple_unoptimized.py - Multiple unoptimized techniques (baseline).

Combines multiple inefficiencies: FP32, small batch, no graphs, unfused ops.
Demonstrates cumulative impact of multiple optimizations.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
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
    BenchmarkMode
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimpleModel(nn.Module):
    """Simple model for optimization demonstration."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineMultipleUnoptimizedBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.x = None
        self.batch_size = 8
        self.hidden_dim = 4096
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).to(torch.float32).eval()
        self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_multiple_unoptimized", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.x)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        try:
            with torch.no_grad():
                test_output = self.model(self.x)
                if test_output.shape[0] != self.batch_size:
                    return f"Output shape mismatch: expected batch_size={self.batch_size}, got {test_output.shape[0]}"
                if test_output.shape[1] != self.hidden_dim:
                    return f"Output shape mismatch: expected hidden_dim={self.hidden_dim}, got {test_output.shape[1]}"
                if not torch.isfinite(test_output).all():
                    return "Output contains non-finite values"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineMultipleUnoptimizedBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=200, warmup=10)
    )
    benchmark = BaselineMultipleUnoptimizedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Multiple Unoptimized Techniques")
    print("=" * 70)
    print("Inefficiencies:")
    print(" 1. FP32 precision (no tensor cores)")
    print(" 2. Small batch size (poor GPU utilization)")
    print(" 3. No CUDA graphs (many kernel launches)")
    print(" 4. Unfused operations (separate kernels)\n")
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")
    print(f"Throughput: {result.iterations / (result.mean_ms * result.iterations / 1000):.2f} iterations/sec")
    print("Status: Multiple inefficiencies combined")
    print("\n Tip: Apply all optimizations for 5-10x cumulative speedup")


if __name__ == "__main__":
    main()
