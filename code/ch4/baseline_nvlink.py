"""baseline_nvlink.py - Multi-GPU baseline without NVLink in distributed training context."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineNVLinkBenchmark(BaseBenchmark):
    """Baseline: PCIe-based communication (no NVLink).
    
    NVLink: This baseline does not use NVLink for high-speed GPU-to-GPU communication.
    Uses PCIe-based communication which is slower.
    """
    
    def __init__(self):
        super().__init__()
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.N = 10_000_000
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
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
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: PCIe-based communication (no NVLink)."""
        with self._nvtx_range("baseline_nvlink"):
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
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data_gpu0 is None:
            return "Data not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineNVLinkBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineNVLinkBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Nvlink")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
