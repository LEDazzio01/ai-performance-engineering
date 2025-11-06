"""baseline_nvlink.py - Baseline without NVLink optimization in storage I/O context.

Demonstrates communication without NVLink optimization.
NVLink: This baseline does not use NVLink for high-speed GPU-to-GPU communication.
Uses PCIe-based communication which is slower.
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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class BaselineNvlinkBenchmark(Benchmark):
    """Baseline: PCIe-based communication (no NVLink).
    
    NVLink: This baseline does not use NVLink for high-speed GPU-to-GPU communication.
    Uses PCIe-based communication which is slower.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: PCIe-based communication (no NVLink)
        # NVLink provides high-speed GPU-to-GPU communication
        # This baseline uses PCIe (slower)
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            # Single GPU: simulate PCIe round-trip
            self.data_gpu0 = torch.randn(self.N, device=self.device, dtype=torch.float32)
            self.data_gpu1 = None
        else:
            # Multi-GPU: use PCIe (not NVLink)
            self.data_gpu0 = torch.randn(self.N, device=torch.device("cuda:0"), dtype=torch.float32)
            self.data_gpu1 = torch.randn(self.N, device=torch.device("cuda:1"), dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: PCIe-based communication (no NVLink)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nvlink", enable=enable_nvtx):
            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                # Multi-GPU: PCIe-based transfer (no NVLink)
                # Transfer through PCIe bus (slower than NVLink)
                self.data_gpu1.copy_(self.data_gpu0, non_blocking=False)
                torch.cuda.synchronize()
            else:
                # Single GPU: simulate inefficient CPU round-trip (no NVLink)
                cpu_data = self.data_gpu0.cpu()
                cpu_data = cpu_data * 2.0  # Simulate CPU operation
                self.data_gpu0 = cpu_data.to(self.device)
                torch.cuda.synchronize()
            
            # Baseline: No NVLink benefits
            # PCIe-based communication (slower)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data_gpu0 = None
        self.data_gpu1 = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data_gpu0 is None:
            return "Data not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineNvlinkBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineNvlinkBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Nvlink")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
