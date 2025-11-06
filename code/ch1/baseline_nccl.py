"""baseline nccl - Baseline without NCCL optimization. Implements Benchmark protocol for harness integration."""

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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineNcclBenchmark(Benchmark):
    """Baseline: Single GPU computation without NCCL (no multi-GPU communication)."""

    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None

    def setup(self) -> None:
        """Setup: Initialize model."""
        torch.manual_seed(42)
        # Baseline: Single GPU - no multi-GPU communication
        # No NCCL collective operations
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Single GPU computation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nccl", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Single GPU computation
                # No multi-GPU communication - no NCCL
                output = self.model(self.input)
                _ = output.sum()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineNcclBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline NCCL (Single GPU): {result.mean_ms:.3f} ms")
