"""Baseline tcgen05 tiling benchmark using cuBLAS (torch.matmul).

This is the baseline wrapper for the SM100 tcgen05 example: it calls the tcgen05
kernel and then copies the returned tensor into a preallocated output buffer.
The optimized variant avoids this redundant copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.tiling_benchmark_base_tcgen05 import TilingBenchmarkBaseTCGen05


class BaselineTilingBenchmarkTCGen05(TilingBenchmarkBaseTCGen05):
    """Baseline tcgen05 wrapper with an extra output copy."""

    nvtx_label = "baseline_tiling_tcgen05"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None
        result = self.extension.matmul_tiling_tcgen05(self.matrix_a, self.matrix_b)
        self.output.copy_(result)


def get_benchmark() -> BaselineTilingBenchmarkTCGen05:
    return BaselineTilingBenchmarkTCGen05()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
