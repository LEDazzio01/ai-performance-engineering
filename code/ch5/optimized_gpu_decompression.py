"""optimized_gpu_decompression.py - GPU-assisted decompression stand-in.

Simulates nvCOMP-style GPU decompression by inflating a toy run-length encoded
buffer on the GPU. Falls back to SKIPPED when CUDA is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


def _encode_rle(length: int = 1024, value: int = 7) -> torch.Tensor:
    """Create a trivial RLE buffer: [run_length, value] pairs."""
    runs = torch.tensor([[length, value]], dtype=torch.int32)
    return runs


class GPUDecompressionBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.encoded: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for GPU decompression demo")
        self.encoded = _encode_rle().to(self.device)
        torch.cuda.synchronize(self.device)

    def _decode_rle(self, runs: torch.Tensor) -> torch.Tensor:
        counts = runs[:, 0].to(torch.int64)
        values = runs[:, 1].to(torch.float32)
        expanded = torch.repeat_interleave(values, counts)
        return expanded

    def benchmark_fn(self) -> Optional[dict]:
        if self.encoded is None:
            raise RuntimeError("SKIPPED: no encoded buffer available")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("gpu_decompress_rle", enable=enable_nvtx):
            out = self._decode_rle(self.encoded)
        torch.cuda.synchronize(self.device)
        latency_ms = self._record_stop(start)
        return {"latency_ms": latency_ms, "output_len": int(out.numel())}

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
    return GPUDecompressionBenchmark()
