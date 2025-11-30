#!/usr/bin/env python3
"""Baseline: Triton matmul with small tile configuration.

Uses 64x64x32 blocks with 4 warps - smaller tiles that have lower
compute throughput but high occupancy.
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

SCHEDULE = MatmulSchedule(
    name="bm64_bn64_bk32",
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    notes="Small tile baseline - high occupancy but lower compute density.",
)


class BaselineProtonMatmul(TritonMatmulProtonBenchmark):
    """Baseline Triton matmul with small tile schedule.
    
    Block config: 64x64x32, 4 warps
    Characteristic: High occupancy but lower compute throughput per block.
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
    return BaselineProtonMatmul()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    if result.timing:
        print(f"\nBaseline Triton Matmul ({SCHEDULE.name})")
        print(f"  Time: {result.timing.mean_ms:.3f} ms")
        print(f"  Notes: {SCHEDULE.notes}")

