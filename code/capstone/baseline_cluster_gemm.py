"""Benchmark wrapper for the capstone baseline GEMM kernel."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from capstone import baseline_matmul
from capstone.capstone_benchmarks import CapstoneMatmulBenchmark


class BaselineCapstoneGemmBenchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        super().__init__(
            runner=baseline_matmul,
            label="capstone_baseline",
            iterations=1,
            warmup=1,
            timeout_seconds=180,
            validate_against_baseline=False,
        )


def get_benchmark() -> BaselineCapstoneGemmBenchmark:
    return BaselineCapstoneGemmBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nCapstone baseline GEMM: "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
