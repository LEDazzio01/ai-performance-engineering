"""Optimized NCCL-style reduction using CUDA ring kernel."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.nccl_benchmark_base import NcclBenchmarkBase


class OptimizedNcclBenchmark(NcclBenchmarkBase):
    nvtx_label = "optimized_nccl"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.device_chunks is not None
        assert self.output is not None
        self.extension.nccl_ring_reduce(self.device_chunks, self.output)


def get_benchmark() -> NcclBenchmarkBase:
    return OptimizedNcclBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = OptimizedNcclBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Optimized NCCL")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()

