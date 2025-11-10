"""baseline_warp_divergence.py - Baseline with warp divergence in GEMM context.

Demonstrates operations that cause warp divergence.
Warp divergence: This baseline has warp divergence issues.
Threads within a warp take different execution paths.
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
from ch10.workload_config import WORKLOAD, is_smoke_test


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineWarpDivergenceBenchmark(Benchmark):
    """Baseline: Operations with warp divergence.
    
    Warp divergence: This baseline has warp divergence issues.
    Threads within a warp take different execution paths, reducing efficiency.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.N = self.workload.warp_elements_for_mode(self.smoke_test)
        self.branch_iterations = self.workload.warp_branch_iterations_for_mode()
        self.input = None
        self.output = None
        self.routing_logits = None
        self._checksum = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.routing_logits = torch.randn(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with warp divergence."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_warp_divergence", enable=enable_nvtx):
            result = self.input.clone()
            mask_source = self.routing_logits
            for iteration in range(self.branch_iterations):
                activations = torch.sigmoid(mask_source)
                mask = activations > 0.5
                positive = result[mask]
                negative = result[~mask]

                positive = torch.tanh(positive * 1.13 + 0.20)
                positive = positive * 1.002 + 0.0006 * positive * positive

                negative = torch.sin(negative * 0.79 - 0.33)
                negative = negative * 0.998 - 0.00035 * negative * negative

                result[mask] = positive
                result[~mask] = negative
                mask_source = 0.9 * mask_source + 0.1 * torch.roll(result, shifts=iteration + 1, dims=0)
                torch.cuda.synchronize()

            self.output = result
            self.routing_logits = mask_source
            self._checksum = float(result.sum().item())

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineWarpDivergenceBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineWarpDivergenceBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Warp Divergence")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
