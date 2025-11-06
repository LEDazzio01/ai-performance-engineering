"""baseline_nccl.py - Baseline without NCCL in GEMM context.

Demonstrates operations without NCCL for multi-GPU communication.
NCCL: This baseline does not use NCCL for collective communication.
Uses CPU-based or inefficient communication patterns.
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
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineNcclBenchmark(Benchmark):
    """Baseline: No NCCL - CPU-based or inefficient communication.
    
    NCCL: This baseline does not use NCCL for collective communication.
    Uses CPU-based or inefficient communication patterns.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.output = None
    
    def setup(self) -> None:
        """Setup: Initialize model without NCCL."""
        torch.manual_seed(42)
        # Baseline: No NCCL - CPU-based communication
        # NCCL provides optimized GPU-to-GPU collective communication
        # This baseline does not use NCCL
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.input = torch.randn(32, 1024, device=self.device)
        self.output = torch.zeros_like(self.input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without NCCL."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nccl", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: No NCCL - CPU-based communication
                # Simulates inefficient communication without NCCL
                output = self.model(self.input)
                
                # Simulate CPU-based communication (no NCCL)
                # Inefficient: requires CPU round-trip
                cpu_output = output.cpu()
                cpu_output = cpu_output * 2.0  # Simulate CPU operation
                self.output = cpu_output.to(self.device)
                
                # Baseline: No NCCL benefits
                # CPU-based communication (inefficient)

    
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
    print(f"Baseline: Nccl")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
