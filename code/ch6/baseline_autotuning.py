"""baseline_autotuning.py - Baseline without autotuning.

Demonstrates operations with fixed parameters (no autotuning).
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
import torch.nn.functional as F
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class BaselineAutotuningBenchmark(Benchmark):
    """Baseline: Fixed parameters without autotuning."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 4_000_000
        self.block_size = 2048  # Fixed micro-chunk
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: Fixed parameters (no autotuning)
        # Autotuning automatically finds optimal kernel parameters
        # This baseline uses fixed block size without tuning
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.mul(1.75)
        out = out.add(0.1)
        return F.silu(out)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with fixed parameters."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_autotuning", enable=enable_nvtx):
            for start in range(0, self.N, self.block_size):
                end = min(start + self.block_size, self.N)
                window = self.input[start:end]
                transformed = self._transform(window)
                self.output[start:end].copy_(transformed)
            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineAutotuningBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Autotuning (Fixed): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Note: Uses fixed parameters, not autotuned for optimal performance")
