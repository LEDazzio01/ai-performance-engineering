"""Optimized tiling benchmark that reuses shared-memory tiles."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.tiling_benchmark_base import TilingBenchmarkBase


class OptimizedTilingBenchmark(TilingBenchmarkBase):
    """Optimized implementation that loads tiles into shared memory."""

    nvtx_label = "optimized_tiling"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None
        if hasattr(self.extension, "matmul_tiled_fast"):
            self.extension.matmul_tiled_fast(self.matrix_a, self.matrix_b, self.output)
        else:
            self.extension.matmul_tiled(self.matrix_a, self.matrix_b, self.output)



def get_benchmark() -> TilingBenchmarkBase:
    """Factory function for harness discovery."""
    return OptimizedTilingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
