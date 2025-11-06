"""baseline_memory_standard.py - Standard memory access baseline (baseline).

Standard memory access patterns without HBM3e optimizations.
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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class BaselineMemoryStandardBenchmark(Benchmark):
    """Standard memory access baseline - no HBM3e optimizations."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.result = None
        self.size_mb = 100  # 100 MB
        self.access_pattern = "sequential"  # Standard sequential access
    
    def setup(self) -> None:
        """Setup: Allocate memory and prepare data."""
        torch.manual_seed(42)
        
        # Allocate standard memory (not optimized for HBM3e)
        num_elements = (self.size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        self.data = torch.randn(num_elements, device=self.device, dtype=torch.float32)
        self.result = torch.zeros_like(self.data)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - standard memory access."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_memory_standard", enable=enable_nvtx):
            # Standard sequential memory access (not optimized for HBM3e)
            # Simple element-wise operations
            self.result = self.data * 2.0 + 1.0
            # Force memory write
            self.result += 0.1

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.data, self.result
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryStandardBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Standard Memory: {result.mean_ms:.3f} ms")

