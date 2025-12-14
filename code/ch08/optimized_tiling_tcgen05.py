"""Optimized tiling benchmark that targets tcgen05 tensor cores."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.tiling_benchmark_base_tcgen05 import TilingBenchmarkBaseTCGen05


class OptimizedTilingBenchmarkTCGen05(TilingBenchmarkBaseTCGen05):
    """Runs the SM100 tcgen05 GEMM."""

    nvtx_label = "optimized_tiling_tcgen05"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        result = self.extension.matmul_tiling_tcgen05(self.matrix_a, self.matrix_b)
        self.output = result



def get_benchmark() -> OptimizedTilingBenchmarkTCGen05:
    return OptimizedTilingBenchmarkTCGen05()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
