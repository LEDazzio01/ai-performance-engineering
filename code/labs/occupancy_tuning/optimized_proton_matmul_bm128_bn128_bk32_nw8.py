#!/usr/bin/env python3
"""Optimized: Triton matmul with high warp count for latency hiding.

Uses 128x128x32 blocks with 8 warps for better latency hiding.
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
    name="bm128_bn128_bk32_nw8",
    block_m=128,
    block_n=128,
    block_k=32,
    num_warps=8,
    notes="More warps for better latency hiding; good for memory-bound scenarios.",
)


class OptimizedProtonMatmulWarpHeavy(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with increased warp count.
    
    Block config: 128x128x32, 8 warps
    Benefit: Better latency hiding via higher warp occupancy.
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
    return OptimizedProtonMatmulWarpHeavy()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    if result.timing:
        print(f"\nWarp-Heavy Triton Matmul ({SCHEDULE.name})")
        print(f"  Time: {result.timing.mean_ms:.3f} ms")
        print(f"  Notes: {SCHEDULE.notes}")

