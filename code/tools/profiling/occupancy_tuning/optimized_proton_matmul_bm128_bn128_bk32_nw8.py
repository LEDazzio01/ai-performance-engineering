"""Optimized target: bm=128, bn=128, bk=32, num_warps=8 (warp-heavy)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.occupancy_tuning.triton_matmul_schedules import (
    WARP_HEAVY_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class OptimizedProtonMatmulBM128BN128BK32NW8Benchmark(TritonMatmulProtonBenchmark):
    def __init__(self) -> None:
        super().__init__(schedule=WARP_HEAVY_SCHEDULE, iterations=2, warmup=1)


def get_benchmark() -> OptimizedProtonMatmulBM128BN128BK32NW8Benchmark:
    return OptimizedProtonMatmulBM128BN128BK32NW8Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nProton matmul (bm128_bn128_bk32_nw8): {mean_ms:.3f} ms")
