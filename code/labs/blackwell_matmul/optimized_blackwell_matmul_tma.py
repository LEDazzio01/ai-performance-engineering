"""Part 2: hardware feature port using shared-memory pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.blackwell_matmul import (
    baseline_blackwell_matmul,
    optimized_blackwell_matmul_pseudo,
    optimized_blackwell_matmul_tma,
)
from labs.blackwell_matmul.blackwell_benchmarks import (
    FeatureDescriptor,
    GraceBlackwellMatmulBenchmark,
)


class TmaGraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        descriptor = FeatureDescriptor(
            tag="tma",
            notes="Part 2: real TMA path (fails fast if TMA unsupported)",
        )
        super().__init__(
            runner=optimized_blackwell_matmul_tma,
            label="grace_blackwell_matmul_tma",
            size=size,
            iterations=5,
            warmup=5,
            descriptor=descriptor,
            reference_runner=optimized_blackwell_matmul_pseudo,
        )
        self.required_capabilities = {"tma": True}


def get_benchmark(size: int = 2048) -> GraceBlackwellMatmulBenchmark:
    return TmaGraceBlackwellBenchmark(size=size)


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nCapstone2 optimized (Part 2 TMA) : {mean_ms:.3f} ms")
