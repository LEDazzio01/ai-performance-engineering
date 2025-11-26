"""baseline_cpu_decompression.py - CPU-bound decompression baseline.

Expands a small buffer compressed with zlib entirely on the CPU. Serves as a
baseline for the GPU-oriented nvCOMP-style path.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class CPUDecompressionBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.compressed: Optional[bytes] = None
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)

    def setup(self) -> None:
        payload = torch.randn(1024 * 1024, dtype=torch.float32).numpy().tobytes()
        self.compressed = zlib.compress(payload, level=6)

    def benchmark_fn(self) -> Optional[dict]:
        if self.compressed is None:
            raise RuntimeError("SKIPPED: no compressed payload available")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("cpu_decompress", enable=enable_nvtx):
            _ = zlib.decompress(self.compressed)
        latency_ms = self._record_stop(start)
        return {"latency_ms": latency_ms, "compressed_bytes": len(self.compressed)}

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

def get_benchmark() -> BaseBenchmark:
    return CPUDecompressionBenchmark()
