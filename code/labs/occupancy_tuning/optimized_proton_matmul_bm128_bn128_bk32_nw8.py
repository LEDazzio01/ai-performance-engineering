#!/usr/bin/env python3
"""Optimized: Triton matmul with high warp count for latency hiding.

Uses 128x128x32 blocks with 8 warps - doubles warp count to improve
latency hiding when register pressure allows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from labs.occupancy_tuning.triton_matmul_schedules import (
    WARP_HEAVY_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class OptimizedProtonMatmulWarpHeavy(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with increased warp count.
    
    Block config: 128x128x32, 8 warps
    Tradeoff: Higher warp count can exacerbate register pressure but
              improves latency hiding when resources allow.
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=WARP_HEAVY_SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmulWarpHeavy()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    if result.timing:
        print(f"\nWarp-Heavy Triton Matmul ({WARP_HEAVY_SCHEDULE.name})")
        print(f"  Time: {result.timing.mean_ms:.3f} ms")
        print(f"  Notes: {WARP_HEAVY_SCHEDULE.notes}")

