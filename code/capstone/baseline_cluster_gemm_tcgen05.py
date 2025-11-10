"""Inline tcgen05 baseline benchmark that mirrors the optimized build."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from capstone import baseline_matmul_non_tcgen05
from capstone.capstone_benchmarks import CapstoneMatmulBenchmark
from capstone.gpu_requirements import ensure_tcgen05_supported


class BaselineCapstoneGemmTCGen05Benchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        ensure_tcgen05_supported()
        super().__init__(
            runner=baseline_matmul_non_tcgen05,
            label="capstone_baseline_tcgen05_inline",
            iterations=1,
            warmup=1,
            timeout_seconds=180,
            validate_against_baseline=False,
        )


def get_benchmark() -> BaselineCapstoneGemmTCGen05Benchmark:
    return BaselineCapstoneGemmTCGen05Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nCapstone baseline tcgen05 GEMM (inline preview): "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
