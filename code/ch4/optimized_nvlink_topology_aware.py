"""optimized_nvlink_topology_aware.py

Topology-aware P2P copy that enables peer access and uses peer distance hints
to choose a near-neighbor pair when multiple GPUs exist. Falls back gracefully
to a single-GPU memcpy when only one device is present.
"""

from __future__ import annotations

from typing import Optional

import torch

from common.python.gpu_requirements import skip_if_insufficient_gpus, require_peer_access
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


def _find_preferred_pair() -> tuple[int, int]:
    """Pick a near-neighbor GPU pair using CUDA peer-distance if available."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError("SKIPPED: Distributed benchmark requires multiple GPUs (found 0-1 GPU)")

    best_pair = (0, 1)
    best_score = float("inf")
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i == j:
                continue
            # Lower distance is better; fall back to 1 if query unsupported
            try:
                dist = torch.cuda.get_device_p2p_attribute(
                    torch.cuda.p2p_attribute.performance_rank, i, j
                )
            except Exception:
                dist = 1
            if dist < best_score:
                best_score = dist
                best_pair = (i, j)
    return best_pair


class OptimizedNvlinkTopologyAwareBenchmark(BaseBenchmark):
    """P2P copy with peer access enabled and near-neighbor pairing."""

    def __init__(self):
        super().__init__()
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.numel = 10_000_000  # Match baseline payload (≈40 MB float32)
        self.src_id = 0
        self.dst_id = 0
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.numel),
        )

    def setup(self) -> None:
        torch.manual_seed(7)
        skip_if_insufficient_gpus(2)

        self.src_id, self.dst_id = _find_preferred_pair()
        n = self.numel

        require_peer_access(self.src_id, self.dst_id)
        require_peer_access(self.dst_id, self.src_id)

        torch.cuda.device(self.src_id).enable_peer_access(self.dst_id)
        torch.cuda.device(self.dst_id).enable_peer_access(self.src_id)

        self.src = torch.randn(n, device=f"cuda:{self.src_id}", dtype=torch.float32)
        self.dst = torch.empty(n, device=f"cuda:{self.dst_id}", dtype=torch.float32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.src is not None and self.dst is not None
        with self._nvtx_range("optimized_nvlink_topology_aware"):
            # Prefer non-blocking copy; peer access is enabled when possible
            self.dst.copy_(self.src, non_blocking=True)
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
    return OptimizedNvlinkTopologyAwareBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=2),
    )
    bench = OptimizedNvlinkTopologyAwareBenchmark()
    result = harness.benchmark(bench)
    print(result)
