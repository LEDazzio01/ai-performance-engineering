"""baseline_launch_bounds.py - Kernel without launch bounds (baseline).

Demonstrates kernel execution without launch bounds annotation.
Uses PyTorch CUDA extension for accurate GPU timing with CUDA Events.
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
    BenchmarkHarness,
    BenchmarkMode,
)

# Import CUDA extension
from ch6.cuda_extensions import load_launch_bounds_extension


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class BaselineLaunchBoundsBenchmark(Benchmark):
    """Kernel without launch bounds annotation (baseline)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input_data = None
        self.output_data = None
        self.N = 1024 * 1024  # 1M elements
        self.iterations = 5
        self._extension = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        self._extension = load_launch_bounds_extension()
        
        torch.manual_seed(42)
        self.input_data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self.output_data = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Kernel without launch bounds."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_launch_bounds", enable=enable_nvtx):
            # Call CUDA extension
            self._extension.launch_bounds_baseline(self.input_data, self.output_data, self.iterations)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input_data = None
        self.output_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input_data is None or self.output_data is None:
            return "Data tensors not initialized"
        if self.input_data.shape[0] != self.N or self.output_data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.output_data).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineLaunchBoundsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Launch Bounds (no annotation): {result.mean_ms:.3f} ms")

