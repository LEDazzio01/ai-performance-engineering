"""Python harness wrapper for tma_multicast_cluster.cu (Chapter 10).

This benchmark participates in the harness as a *comparable* optimized variant
for the cluster multicast example.

Pairs with: baseline_cluster_multicast.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedClusterMulticastBenchmark(CudaBinaryBenchmark):
    """Runs tma_multicast_cluster.cu (cluster multicast)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="tma_multicast_cluster",
            friendly_name="TMA Multicast Optimized (Cluster Multicast)",
            iterations=1,
            warmup=1,
            timeout_seconds=120,
            workload_params={
                "batch_size": 1,
                "dtype": "float32",
                "M": 2048,
                "N": 2048,
                "K": 2048,
                "tile_m": 64,
                "tile_n": 64,
                "tile_k": 32,
                "cluster_m": 4,
                "cluster_n": 1,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=48 * 1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_bandwidth_metrics
        if self.last_time_ms is None:
            raise RuntimeError("Benchmark did not capture TIME_MS output")
        total_bytes = 48 * 1024 * 1024
        return compute_bandwidth_metrics(total_bytes=total_bytes, elapsed_ms=float(self.last_time_ms))

    def get_input_signature(self) -> dict:
        return simple_signature(batch_size=1, dtype="float32", M=2048, N=2048, K=2048).to_dict()


def get_benchmark() -> BaseBenchmark:
    return OptimizedClusterMulticastBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

