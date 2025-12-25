"""Python harness wrapper for tma_multicast_baseline.cu (Chapter 10).

This benchmark participates in the harness as a *comparable* baseline for the
cluster multicast optimization.

Pairs with: optimized_cluster_multicast.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BaseBenchmark


class BaselineClusterMulticastBenchmark(CudaBinaryBenchmark):
    """Runs tma_multicast_baseline.cu (no cluster multicast)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        m = 4096
        n = 4096
        k = 4096
        bytes_per_iteration = (m * k + k * n + m * n) * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="tma_multicast_baseline",
            friendly_name="TMA Multicast Baseline (No Cluster Multicast)",
            iterations=1,
            warmup=1,
            timeout_seconds=120,
            run_args=[str(m), str(n), str(k)],
            workload_params={
                "batch_size": 1,
                "dtype": "float32",
                "M": m,
                "N": n,
                "K": k,
                "tile_m": 4,
                "tile_n": 128,
                "tile_k": 128,
                "cluster_m": 16,
                "cluster_n": 1,
            },
        )
        self._m = m
        self._n = n
        self._k = k
        self._bytes_per_iteration = bytes_per_iteration
        self.register_workload_metadata(bytes_per_iteration=bytes_per_iteration)

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_bandwidth_metrics
        if self.last_time_ms is None:
            raise RuntimeError("Benchmark did not capture TIME_MS output")
        return compute_bandwidth_metrics(
            total_bytes=self._bytes_per_iteration,
            elapsed_ms=float(self.last_time_ms),
        )

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=1,
            dtype="float32",
            M=self._m,
            N=self._n,
            K=self._k,
        ).to_dict()


def get_benchmark() -> BaseBenchmark:
    return BaselineClusterMulticastBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
