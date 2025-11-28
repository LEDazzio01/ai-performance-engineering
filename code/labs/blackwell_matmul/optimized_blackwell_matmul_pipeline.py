"""Part 3: multi-stage accumulation reaching 85%+ of SOTA."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.blackwell_matmul import (
    baseline_blackwell_matmul,
    optimized_blackwell_matmul_pipeline,
)
from labs.blackwell_matmul.blackwell_benchmarks import (
    FeatureDescriptor,
    GraceBlackwellMatmulBenchmark,
)


def _maybe_compile_runner():
    try:
        return torch.compile(
            optimized_blackwell_matmul_pipeline,
            mode="max-autotune",
            fullgraph=False,
        )
    except Exception:
        # Torch.compile may be unavailable if torch was built without dynamo.
        return optimized_blackwell_matmul_pipeline


_PIPELINE_RUNNER = _maybe_compile_runner()


class PipelineGraceBlackwellBenchmark(GraceBlackwellMatmulBenchmark):
    def __init__(self, size: int = 2048) -> None:
        descriptor = FeatureDescriptor(
            tag="pipeline",
            notes="Part 3: prefetch distance sweep + PyTorch 2.10 compile() glue",
        )
        super().__init__(
            runner=_PIPELINE_RUNNER,
            label="grace_blackwell_matmul_pipeline",
            size=size,
            iterations=7,
            warmup=10,
            descriptor=descriptor,
            reference_runner=baseline_blackwell_matmul,
        )
        self.required_capabilities = {}


def get_benchmark(size: int = 2048) -> GraceBlackwellMatmulBenchmark:
    return PipelineGraceBlackwellBenchmark(size=size)


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nCapstone2 optimized (Part 3 pipeline) : {mean_ms:.3f} ms")
