#!/usr/bin/env python3
"""Optimized: Triton matmul with wide-N tile for bandwidth-bound cases.

Uses 128x256x64 blocks - wide N dimension to highlight shared memory
pressure effects on theoretical occupancy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from labs.occupancy_tuning.triton_matmul_schedules import (
    EXTRA_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class OptimizedProtonMatmulWideN(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with wide-N tile configuration.
    
    Block config: 128x256x64, 4 warps
    Use case: Highlights when shared memory pressure caps theoretical
              occupancy. Good for memory bandwidth experiments.
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=EXTRA_SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmulWideN()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    if result.timing:
        print(f"\nWide-N Triton Matmul ({EXTRA_SCHEDULE.name})")
        print(f"  Time: {result.timing.mean_ms:.3f} ms")
        print(f"  Notes: {EXTRA_SCHEDULE.notes}")

