#!/usr/bin/env python3
"""Optimized: Triton matmul with minimal resource footprint.

Uses 64x64x32 blocks with 2 warps - minimal per-block resources
for near-100% theoretical occupancy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from labs.occupancy_tuning.triton_matmul_schedules import (
    LATENCY_FRIENDLY_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class OptimizedProtonMatmulLatencyFriendly(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with minimal resource footprint.
    
    Block config: 64x64x32, 2 warps
    Benefit: Small tiles minimize per-block resources so theoretical
             occupancy approaches 100%. Good for latency-sensitive workloads.
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=LATENCY_FRIENDLY_SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmulLatencyFriendly()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    if result.timing:
        print(f"\nLatency-Friendly Triton Matmul ({LATENCY_FRIENDLY_SCHEDULE.name})")
        print(f"  Time: {result.timing.mean_ms:.3f} ms")
        print(f"  Notes: {LATENCY_FRIENDLY_SCHEDULE.notes}")

