#!/usr/bin/env python3
"""Baseline locality microbench (host-pinned -> GPU copy each iteration).

Simulates “remote” placement by copying from pinned host memory every iteration
before doing trivial math. Optimized pair keeps data resident on device.
Works on any CUDA GPU; label mirrors Grace-Blackwell locality story.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineGb200LocalityBenchmark(BaseBenchmark):
    def __init__(self, size_mb: float = 256.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((self.size_mb * 1024 * 1024) / 4)  # float32
        self.host_buf: Optional[torch.Tensor] = None
        self.device_buf: Optional[torch.Tensor] = None
        tokens = float(self.numel)
        self._workload = WorkloadMetadata(tokens_per_iteration=tokens, requests_per_iteration=1.0)

    def setup(self) -> None:
        # Pinned host buffer to simulate “remote” access each iteration.
        self.host_buf = torch.ones(self.numel, dtype=torch.float32, pin_memory=True)
        # Destination buffer on device.
        self.device_buf = torch.empty_like(self.host_buf, device=self.device, pin_memory=False)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        assert self.host_buf is not None and self.device_buf is not None
        with self._nvtx_range("host_to_device+compute"):
            self.device_buf.copy_(self.host_buf, non_blocking=True)
            self.device_buf.add_(1.0)
        self._synchronize()

    def teardown(self) -> None:
        self.host_buf = None
        self.device_buf = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5, enable_memory_tracking=False, enable_profiling=False)

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
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineGb200LocalityBenchmark()
