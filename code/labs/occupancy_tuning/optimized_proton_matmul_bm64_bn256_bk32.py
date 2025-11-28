#!/usr/bin/env python3
"""Optimized: Triton matmul with higher-occupancy tile configuration.

Uses 64x256x32 blocks - trims registers per thread and achieves higher active warps.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from labs.occupancy_tuning.triton_matmul_schedules import (
    OPTIMIZED_SCHEDULE,
    TritonMatmulProtonBenchmark,
)


class OptimizedProtonMatmul(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with higher-occupancy schedule.
    
    Block config: 64x256x32, 4 warps
    Benefit: Lower registers/thread = higher occupancy = better latency hiding
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=OPTIMIZED_SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmul()


if __name__ == "__main__":
    from labs.occupancy_tuning.baseline_proton_matmul import BaselineProtonMatmul
    
    # Run baseline
    baseline = BaselineProtonMatmul()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=baseline.get_config())
    baseline_result = harness.benchmark(baseline)
    baseline_time = baseline_result.timing.mean_ms if baseline_result.timing else 0
    
    # Run optimized
    optimized = get_benchmark()
    harness2 = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=optimized.get_config())
    optimized_result = harness2.benchmark(optimized)
    optimized_time = optimized_result.timing.mean_ms if optimized_result.timing else 0
    
    print(f"\n=== Occupancy Tuning Results ===")
    print(f"Baseline ({baseline.schedule.name}): {baseline_time:.3f} ms")
    print(f"Optimized ({OPTIMIZED_SCHEDULE.name}): {optimized_time:.3f} ms")
    if baseline_time > 0:
        print(f"Speedup: {baseline_time / optimized_time:.2f}x")
    print(f"\nOptimization: {OPTIMIZED_SCHEDULE.notes}")

