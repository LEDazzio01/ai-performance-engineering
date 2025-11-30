"""baseline_memory_transfer.py - Traditional PCIe memory transfer (baseline)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineMemoryTransferBenchmark(BaseBenchmark):
    """Traditional PCIe memory transfer - slower path."""
    
    def __init__(self):
        super().__init__()
        self.host_data: Optional[torch.Tensor] = None
        self.device_data: Optional[torch.Tensor] = None
        self.N = 10_000_000
        bytes_per_iter = self.N * 4  # float32 copy
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self.host_data = torch.randn(self.N, dtype=torch.float32, pin_memory=True)
        self.device_data = torch.empty(self.N, dtype=torch.float32, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Traditional H2D transfer over PCIe."""
        assert self.host_data is not None and self.device_data is not None
        with self._nvtx_range("memory_transfer_baseline"):
            self.device_data.copy_(self.host_data, non_blocking=False)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.device_data is None:
            return "Device tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryTransferBenchmark()
