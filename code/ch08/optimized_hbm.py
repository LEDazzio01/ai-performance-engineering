"""HBM optimized benchmark with vectorized, contiguous access."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.hbm_benchmark_base import HBMBenchmarkBase


class OptimizedHBMBenchmark(HBMBenchmarkBase):
    nvtx_label = "optimized_hbm"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_row is not None
        assert self.output is not None
        self.extension.hbm_optimized(self.matrix_row, self.output)



def get_benchmark() -> HBMBenchmarkBase:
    return OptimizedHBMBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
