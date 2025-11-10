"""Baseline capstone scenario: kernel optimization."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capstone.scenario_benchmark import CapstoneScenarioBenchmark, ScenarioVariant


class BaselineKernelOptimizationBenchmark(CapstoneScenarioBenchmark):
    def __init__(self) -> None:
        super().__init__("04_kernel_optimization", ScenarioVariant.BASELINE)


def get_benchmark() -> CapstoneScenarioBenchmark:
    return BaselineKernelOptimizationBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nCapstone baseline kernel optimization: {mean_ms:.3f} ms")
