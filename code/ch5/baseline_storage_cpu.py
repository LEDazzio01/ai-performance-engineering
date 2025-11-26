"""baseline_storage_cpu.py - CPU-mediated storage I/O (baseline)."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineStorageCpuBenchmark(BaseBenchmark):
    """CPU-mediated I/O - double copy overhead (storage→CPU→GPU)."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.filepath: Optional[str] = None
        self.size_mb = 64  # Smaller for faster benchmark
        self.size = self.size_mb * 1024 * 1024 // 4  # float32 elements
        bytes_per_iter = self.size * 4 * 2  # write + read
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize data and create temp file."""
        torch.manual_seed(42)
        self.data = torch.randn(self.size, device=self.device, dtype=torch.float32)
        
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        self.filepath = f.name
        f.close()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: CPU-mediated I/O."""
        assert self.data is not None and self.filepath is not None
        with self._nvtx_range("storage_cpu"):
            cpu_data = self.data.cpu().numpy()
            np.save(self.filepath, cpu_data)
            cpu_loaded = np.load(self.filepath)
            self.data = torch.from_numpy(cpu_loaded).to(self.device)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.filepath and os.path.exists(self.filepath):
            os.unlink(self.filepath)
        self.data = None
        self.filepath = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.size:
            return f"Data size mismatch: expected {self.size}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineStorageCpuBenchmark()
