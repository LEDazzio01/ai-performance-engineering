"""baseline_hbm.py - Baseline without HBM optimization in kernel efficiency/arithmetic intensity context.

Demonstrates operations without HBM memory optimization.
HBM: This baseline does not optimize for HBM memory bandwidth.
Uses standard memory access patterns without HBM-specific optimizations.
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
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class BaselineHbmBenchmark(Benchmark):
    """Baseline: Standard memory access without HBM optimization.
    
    HBM: This baseline does not optimize for HBM memory bandwidth.
    Uses standard memory access patterns without HBM-specific optimizations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self._last = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model without HBM optimization."""
        torch.manual_seed(42)
        # Baseline: Standard memory access (no HBM optimization)
        # HBM (High Bandwidth Memory) provides high memory bandwidth
        # This baseline does not optimize for HBM
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()

        torch.manual_seed(42)
        self.micro_batch = 8
        self.num_micro_batches = 64
        self.micro_batches = [
            torch.randn(self.micro_batch, 1024, device=self.device)
            for _ in range(self.num_micro_batches)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without HBM optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_hbm", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Standard memory access (no HBM optimization)
                # Does not leverage HBM high bandwidth characteristics
                # HBM optimization would use bandwidth-optimized access patterns
                total = 0.0
                for shard in self.micro_batches:
                    result = self.model(shard)
                    total += float(result.sum().item())
                    torch.cuda.synchronize()
                self._last = total

    
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
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineHbmBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineHbmBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: HBM")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
