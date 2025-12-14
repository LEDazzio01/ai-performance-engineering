"""HBM baseline benchmark with poor memory layout."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.hbm_benchmark_base import HBMBenchmarkBase


class BaselineHBMBenchmark(HBMBenchmarkBase):
    nvtx_label = "baseline_hbm"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.host_col is not None
        assert self.matrix_col is not None
        assert self.output is not None
        # Naive path: copy pageable host memory to device every iteration before running the kernel.
        self.matrix_col.copy_(self.host_col, non_blocking=False)
        self.extension.hbm_baseline(self.matrix_col, self.output)



def get_benchmark() -> HBMBenchmarkBase:
    return BaselineHBMBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
