#!/usr/bin/env python3
"""Optimized: Triton matmul with larger tile configuration for Blackwell.

Uses 128x128x64 blocks - better compute density on high-bandwidth Blackwell SM.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.profiling.occupancy_tuning.triton_matmul_schedules import (
    MatmulSchedule,
    TritonMatmulProtonBenchmark,
)

# This benchmark uses the optimized 128x128x64 config 
# that performs well on Blackwell's wide execution resources
SCHEDULE = MatmulSchedule(
    name="bm128_bn128_bk64",
    block_m=128,
    block_n=128,
    block_k=64,
    num_warps=4,
    notes="Larger tile for better compute density on Blackwell SM.",
)


class OptimizedProtonMatmul(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with Blackwell-friendly tile config.
    
    Block config: 128x128x64, 4 warps
    Benefit: Better compute density on high-bandwidth SM
    """

    def __init__(self, size: int = 4096):
        super().__init__(
            schedule=SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
            use_compile=False,  # Avoid torch.compile issues with Triton
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

