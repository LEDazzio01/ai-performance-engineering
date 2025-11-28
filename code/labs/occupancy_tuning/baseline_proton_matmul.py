#!/usr/bin/env python3
"""Baseline: Triton matmul with reference tile configuration.

Uses 128x128x64 blocks with 4 warps - standard configuration that
often has high register pressure and suboptimal occupancy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from labs.occupancy_tuning.triton_matmul_schedules import (
    BASELINE_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class BaselineProtonMatmul(TritonMatmulProtonBenchmark):
    """Baseline Triton matmul with reference schedule.
    
    Block config: 128x128x64, 4 warps
    Known for: High register pressure, often falls below predicted occupancy
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=BASELINE_SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineProtonMatmul()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    if result.timing:
        print(f"\nBaseline Triton Matmul ({BASELINE_SCHEDULE.name})")
        print(f"  Time: {result.timing.mean_ms:.3f} ms")
        print(f"  Notes: {BASELINE_SCHEDULE.notes}")

