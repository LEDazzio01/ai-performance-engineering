"""baseline_nvlink.py - Baseline without NVLink optimization in AI optimization context.

Demonstrates memory transfer without NVLink (uses PCIe).
NVLink: This baseline does not use NVLink for GPU-to-GPU communication.
Uses slower PCIe for transfers, limiting multi-GPU performance.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class BaselineNvlinkBenchmark(Benchmark):
    """Baseline: Memory transfer without NVLink (PCIe only).
    
    NVLink: This baseline does not use NVLink for GPU-to-GPU communication.
    Uses slower PCIe for transfers, limiting multi-GPU performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.host_data = None
        self.device_data = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize host and device memory."""
        torch.manual_seed(42)
        # Baseline: Standard host-device transfer without NVLink
        # NVLink provides high-speed GPU-to-GPU and CPU-to-GPU communication
        # This baseline uses PCIe, which is slower
        
        # Allocate host memory (pinned for faster transfer, but still PCIe)
        self.host_data = torch.randn(self.N, pin_memory=True)
        
        # Allocate device memory
        self.device_data = torch.empty(self.N, device=self.device)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Memory transfer without NVLink."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nvlink", enable=enable_nvtx):
            # Baseline: PCIe transfer (no NVLink)
            # NVLink would provide much higher bandwidth for GPU-to-GPU transfers
            # This baseline uses standard PCIe, which is slower
            
            # Simulate inefficient transfer pattern (e.g., via CPU)
            # In multi-GPU, this would use PCIe instead of NVLink
            temp_cpu = self.host_data.cpu()  # Force CPU round-trip (inefficient)
            
            # Host to device transfer (PCIe-like, via CPU)
            self.device_data.copy_(temp_cpu, non_blocking=False)
            
            # Process on device
            _ = self.device_data * 2.0
            
            # Device to host transfer (PCIe-like, via CPU)
            temp_cpu.copy_(self.device_data, non_blocking=False)
            self.host_data.copy_(temp_cpu, non_blocking=False)
            
            # Baseline: No NVLink optimization
            # Transfer speed limited by PCIe bandwidth and CPU round-trip overhead

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.host_data is None or self.device_data is None:
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
    print(f"Baseline: nvlink")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
