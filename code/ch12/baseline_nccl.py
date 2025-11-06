"""baseline_nccl.py - Baseline without NCCL in kernel launches/CUDA graphs context.

Demonstrates operations without NCCL and many kernel launches (no CUDA graphs).
NCCL: This baseline does not use NCCL for collective communication.
Multiple kernel launches without CUDA graphs optimization.
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
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class BaselineNcclBenchmark(Benchmark):
    """Baseline: No NCCL with many kernel launches (no CUDA graphs).
    
    NCCL: This baseline does not use NCCL for collective communication.
    Multiple kernel launches without CUDA graphs optimization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.output = None
    
    def setup(self) -> None:
        """Setup: Initialize model without NCCL or CUDA graphs."""
        torch.manual_seed(42)
        # Baseline: No NCCL with many kernel launches
        # NCCL: not used for collective communication
        # No CUDA graphs: each operation launches separate kernel
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.input = torch.randn(32, 1024, device=self.device)
        self.output = torch.zeros_like(self.input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without NCCL (many kernel launches)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nccl", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: No NCCL with many kernel launches
                # NCCL: not used for collective communication
                # No CUDA graphs: multiple kernel launches
                output = self.model(self.input)  # Multiple kernel launches
                
                # Simulate CPU-based communication (no NCCL)
                # Inefficient: requires CPU round-trip
                cpu_output = output.cpu()  # Kernel launch
                cpu_output = cpu_output * 2.0  # CPU operation
                self.output = cpu_output.to(self.device)  # Kernel launch
                
                # Baseline: No NCCL or CUDA graphs benefits
                # - CPU-based communication (inefficient)
                # - Many kernel launches (high overhead)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
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
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineNcclBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineNcclBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: NCCL")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

