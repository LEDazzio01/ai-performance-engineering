"""optimized_nvlink.py - Optimized NVLink transfers for distributed training context."""

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


class OptimizedNVLinkBenchmark(BaseBenchmark):
    """Optimized: NVLink for high-speed GPU-to-GPU communication.
    
    NVLink: Uses NVLink for optimized GPU-to-GPU transfers.
    Provides high bandwidth and low latency compared to PCIe.
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
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: NVLink for high-speed GPU-to-GPU communication
        # NVLink provides high bandwidth and low latency
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            # Single GPU: use optimized pinned memory (simulates NVLink benefit)
            self.data_gpu0 = torch.randn(self.N, device=self.device, dtype=torch.float32)
            self.data_gpu1 = None
        else:
            # Multi-GPU: use NVLink for direct GPU-to-GPU transfer
            self.data_gpu0 = torch.randn(self.N, device=torch.device("cuda:0"), dtype=torch.float32)
            self.data_gpu1 = torch.randn(self.N, device=torch.device("cuda:1"), dtype=torch.float32)
            
            # Enable peer access for NVLink (if available)
            if torch.cuda.can_device_access_peer(0, 1):
                torch.cuda.device(0).enable_peer_access(1)
                torch.cuda.device(1).enable_peer_access(0)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: NVLink-optimized communication."""
        with self._nvtx_range("optimized_nvlink"):
            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                # Multi-GPU: NVLink-optimized transfer
                # Direct GPU-to-GPU copy via NVLink (high bandwidth, low latency)
                self.data_gpu1.copy_(self.data_gpu0, non_blocking=True)
                torch.cuda.synchronize()
            else:
                # Single GPU: optimized pinned memory transfer (simulates NVLink benefit)
                # NVLink would provide high-speed transfer in multi-GPU setup
                pinned_data = torch.empty(self.N, dtype=torch.float32, pin_memory=True)
                pinned_data.copy_(self.data_gpu0, non_blocking=True)
                self.data_gpu0 = pinned_data.to(self.device, non_blocking=True)
                torch.cuda.synchronize()
            
            # Optimization: NVLink benefits
            # - High-speed GPU-to-GPU communication
            # - High bandwidth and low latency
            # - Better performance than PCIe
            # - Direct GPU-to-GPU transfer

    
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
    return OptimizedNVLinkBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedNVLinkBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Nvlink")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
