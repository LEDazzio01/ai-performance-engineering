"""Optimized target: bm=64, bn=64, bk=32, num_warps=2 (latency-friendly)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.occupancy_tuning.triton_matmul_schedules import (
    LATENCY_FRIENDLY_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class OptimizedProtonMatmulBM64BN64BK32NW2Benchmark(TritonMatmulProtonBenchmark):
    def __init__(self) -> None:
        super().__init__(schedule=LATENCY_FRIENDLY_SCHEDULE, iterations=2, warmup=1)


def get_benchmark() -> OptimizedProtonMatmulBM64BN64BK32NW2Benchmark:
    return OptimizedProtonMatmulBM64BN64BK32NW2Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nProton matmul (bm64_bn64_bk32_nw2): {mean_ms:.3f} ms")
