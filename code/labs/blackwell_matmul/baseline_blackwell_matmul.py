"""Baseline Grace-Blackwell matmul benchmark (Part 1)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.blackwell_matmul import baseline_blackwell_matmul
from labs.blackwell_matmul.blackwell_benchmarks import (
    FeatureDescriptor,
    GraceBlackwellMatmulBenchmark,
)


class BaselineGraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        descriptor = FeatureDescriptor(
            tag="baseline",
            notes="Part 1 roofline walkthrough with a naïve CUDA kernel",
        )
        super().__init__(
            runner=baseline_blackwell_matmul,
            label="grace_blackwell_matmul_baseline",
            size=size,
            iterations=3,
            warmup=5,
            descriptor=descriptor,
            reference_runner=None,
        )


def get_benchmark(size: int = 2048) -> GraceBlackwellMatmulBenchmark:
    return BaselineGraceBlackwellBenchmark(size=size)


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nCapstone2 baseline (Part 1) : {mean_ms:.3f} ms")
