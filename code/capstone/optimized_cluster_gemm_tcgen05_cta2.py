"""tcgen05 CTA-group::2 benchmark built for SM100 hardware."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from capstone import optimized_matmul_tcgen05_cta2
from capstone.capstone_benchmarks import CapstoneMatmulBenchmark
from capstone.gpu_requirements import ensure_tcgen05_supported


class OptimizedCapstoneGemmTCGen05CTA2Benchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        ensure_tcgen05_supported()
        super().__init__(
            runner=optimized_matmul_tcgen05_cta2,
            label="capstone_optimized_tcgen05_cta2",
            iterations=3,
            warmup=1,
            timeout_seconds=360,
            validate_against_baseline=False,
        )


def get_benchmark() -> OptimizedCapstoneGemmTCGen05CTA2Benchmark:
    return OptimizedCapstoneGemmTCGen05CTA2Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nCapstone optimized tcgen05 CTA2 GEMM: "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
