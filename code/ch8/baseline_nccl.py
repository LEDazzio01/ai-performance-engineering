"""Baseline NCCL reduction using host-side aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch8.nccl_benchmark_base import NcclBenchmarkBase


class BaselineNcclBenchmark(NcclBenchmarkBase):
    nvtx_label = "baseline_nccl"

    def _invoke_kernel(self) -> None:
        assert self.device_chunks is not None
        assert self.host_chunks is not None
        assert self.output is not None

        # Naive path: copy each rank to host, accumulate on CPU, copy result back.
        host_accum = torch.zeros(self.chunk_elems, dtype=torch.float32)
        for rank in range(self.world_size):
            host_accum += self.host_chunks[rank]

        self.output.copy_(host_accum.to(self.output.device))


def get_benchmark() -> NcclBenchmarkBase:
    return BaselineNcclBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = BaselineNcclBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Baseline NCCL")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
