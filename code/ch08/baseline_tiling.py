"""Naive matmul baseline that skips tiling/shared-memory reuse."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.tiling_benchmark_base import TilingBenchmarkBase


class BaselineTilingBenchmark(TilingBenchmarkBase):
    """Baseline implementation: every multiply reads directly from HBM."""

    nvtx_label = "baseline_tiling"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None
        self.extension.matmul_naive(self.matrix_a, self.matrix_b, self.output)



def get_benchmark() -> TilingBenchmarkBase:
    """Factory function for harness discovery."""
    return BaselineTilingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
