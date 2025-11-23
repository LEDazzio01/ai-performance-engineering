"""Part 5: tcgen05 + TMEM + CTA-group variants (Blackwell SM100/SM12x)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.tcgen05_requirements import ensure_tcgen05_supported
from labs.blackwell_matmul import (
    baseline_blackwell_matmul,
    optimized_blackwell_matmul_tcgen05,
    optimized_blackwell_matmul_tcgen05_cta2,
)
from labs.blackwell_matmul.blackwell_benchmarks import (
    FeatureDescriptor,
    GraceBlackwellMatmulBenchmark,
)


class Tcgen05GraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        ensure_tcgen05_supported()
        descriptor = FeatureDescriptor(
            tag="tcgen05_tmem",
            notes="tcgen05 tensor cores + TMEM epilogue (single-CTA per tile)",
        )
        super().__init__(
            runner=optimized_blackwell_matmul_tcgen05,
            label="grace_blackwell_matmul_tcgen05",
            size=size,
            iterations=5,
            warmup=2,
            descriptor=descriptor,
            reference_runner=baseline_blackwell_matmul,
        )
        self.required_capabilities = {"tcgen05": True}


class Tcgen05Cta2GraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        ensure_tcgen05_supported()
        descriptor = FeatureDescriptor(
            tag="tcgen05_cta2",
            notes="tcgen05 + TMEM with CTA-group::2 multicast (cluster launch)",
        )
        super().__init__(
            runner=optimized_blackwell_matmul_tcgen05_cta2,
            label="grace_blackwell_matmul_tcgen05_cta2",
            size=size,
            iterations=5,
            warmup=2,
            descriptor=descriptor,
            reference_runner=baseline_blackwell_matmul,
        )
        self.required_capabilities = {"tcgen05": True, "cta_group": True}


def get_benchmark(size: int = 2048, *, cta2: bool = False) -> GraceBlackwellMatmulBenchmark:
    """Factory for discover_benchmarks()."""
    if cta2:
        return Tcgen05Cta2GraceBlackwellBenchmark(size=size)
    return Tcgen05GraceBlackwellBenchmark(size=size)


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\ntcgen05 (TMEM) : {mean_ms:.3f} ms")
