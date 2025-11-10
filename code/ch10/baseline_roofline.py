"""baseline_roofline.py - Baseline without roofline analysis in GEMM context.

Demonstrates operations without roofline analysis for performance optimization.
Roofline: This baseline does not use roofline analysis.
Does not measure or optimize based on compute/memory bottlenecks.
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


class BaselineRooflineBenchmark(Benchmark):
    """Baseline: Operations without roofline analysis.
    
    Roofline: This baseline does not use roofline analysis.
    Does not measure or optimize based on compute/memory bottlenecks.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs_host = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.batch = self.workload.roofline_batch_for_mode(self.smoke_test)
        self.micro_batches = self.workload.roofline_micro_batches_for_mode(self.smoke_test)
        self.hidden_dim = self.workload.roofline_hidden_dim_for_mode(self.smoke_test)
        self.ffn_dim = self.workload.roofline_ffn_dim_for_mode(self.smoke_test)
        self._checksum = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model without roofline analysis."""
        torch.manual_seed(42)
        # Baseline: No roofline analysis
        # Roofline analysis identifies compute-bound vs memory-bound operations
        # This baseline does not perform roofline analysis
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.hidden_dim),
        ).to(self.device).float().eval()
        
        self.inputs_host = torch.randn(
            self.micro_batches,
            self.batch,
            self.hidden_dim,
            pin_memory=True,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without roofline analysis."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_roofline", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: No roofline analysis
                # Does not measure arithmetic intensity or identify bottlenecks
                # No optimization based on compute/memory characteristics
                total = 0.0
                for idx in range(self.micro_batches):
                    host_chunk = self.inputs_host[idx]
                    device_chunk = host_chunk.to(self.device, non_blocking=False)
                    output = self.model(device_chunk)
                    total += float(output.sum().item())
                    torch.cuda.synchronize()
                self._checksum = total

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs_host = None
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
        if self.inputs_host is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineRooflineBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineRooflineBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Roofline")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
