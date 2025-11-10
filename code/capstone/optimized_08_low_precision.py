"""Optimized capstone scenario: low precision."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capstone.scenario_benchmark import CapstoneScenarioBenchmark, ScenarioVariant


class OptimizedLowPrecisionBenchmark(CapstoneScenarioBenchmark):
    def __init__(self) -> None:
        super().__init__("08_low_precision", ScenarioVariant.OPTIMIZED)


def get_benchmark() -> CapstoneScenarioBenchmark:
    return OptimizedLowPrecisionBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nCapstone optimized low precision: {mean_ms:.3f} ms")
