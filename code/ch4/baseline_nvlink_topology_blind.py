"""baseline_nvlink_topology_blind.py

Baseline NVLink benchmark that ignores topology: default stream, no peer access enablement,
and no attempt to pick near-neighbor pairs. Uses a simple P2P copy between GPU 0 and GPU 1
if available, otherwise falls back to a single-GPU memcpy to keep the harness runnable.
"""

from __future__ import annotations

from typing import Optional

import torch

from common.python.gpu_requirements import skip_if_insufficient_gpus, require_peer_access
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineNvlinkTopologyBlindBenchmark(BaseBenchmark):
    """Naive P2P copy that does not enable peer access or respect NVLink distance."""

    def __init__(self):
        super().__init__()
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.numel = 8 * 1024 * 1024  # 32 MB
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.numel),
        )

    def setup(self) -> None:
        torch.manual_seed(7)
        n = self.numel
        skip_if_insufficient_gpus(2)
        require_peer_access(0, 1)

        self.src = torch.randn(n, device="cuda:0", dtype=torch.float16)
        self.dst = torch.empty(n, device="cuda:1", dtype=torch.float16)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.src is not None and self.dst is not None
        with self._nvtx_range("baseline_nvlink_topology_blind"):
            # Naive: default stream copy, peer access may be disabled
            self.dst.copy_(self.src, non_blocking=False)
            self._synchronize()

    def teardown(self) -> None:
        self.src = None
        self.dst = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

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
        if self.src is None or self.dst is None:
            return "Buffers not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineNvlinkTopologyBlindBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=5),
    )
    bench = BaselineNvlinkTopologyBlindBenchmark()
    result = harness.benchmark(bench)
    print(result)
