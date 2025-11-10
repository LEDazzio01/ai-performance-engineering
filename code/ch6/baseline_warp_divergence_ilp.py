"""baseline_warp_divergence_ilp.py - Baseline ILP with warp divergence.

Demonstrates ILP operations that cause warp divergence, limiting parallelism.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from typing import Optional
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch6.workload_config import WORKLOAD, is_smoke_test


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class BaselineWarpDivergenceILPBenchmark(Benchmark):
    """Baseline: ILP limited by warp divergence."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.routing_logits = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.N = self.workload.warp_elements_for_mode(self.smoke_test)
        self.branch_iterations = self.workload.warp_branch_iterations
        self._checksum = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: ILP operations with warp divergence
        # Warp divergence occurs when threads in a warp take different paths
        # This limits instruction-level parallelism
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.routing_logits = torch.randn(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: ILP operations with warp divergence."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_warp_divergence_ilp", enable=enable_nvtx):
            # Baseline: ILP limited by warp divergence
            # Warp divergence causes serialization of divergent paths
            # This reduces instruction-level parallelism
            # Threads in same warp take different execution paths
            mask_source = self.routing_logits
            result = self.input.clone()
            for iteration in range(self.branch_iterations):
                activations = torch.sigmoid(mask_source)
                mask = activations > 0.5

                positive = result[mask]
                negative = result[~mask]

                positive = torch.tanh(positive * 1.11 + 0.25)
                positive = positive * 1.003 + 0.0005 * positive * positive

                negative = torch.sin(negative * 0.77 - 0.35)
                negative = negative * 0.997 - 0.0004 * negative * negative

                result[mask] = positive
                result[~mask] = negative

                mask_source = 0.92 * mask_source + 0.08 * torch.roll(result, shifts=iteration + 1, dims=0)
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
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineWarpDivergenceILPBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Warp Divergence ILP: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Note: Warp divergence limits instruction-level parallelism")
