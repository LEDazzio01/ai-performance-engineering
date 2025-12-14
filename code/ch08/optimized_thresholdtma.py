"""Optimized threshold benchmark using CUDA pipeline/TMA staging."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.threshold_tma_benchmark_base import ThresholdBenchmarkBaseTMA


class OptimizedThresholdTMABenchmark(ThresholdBenchmarkBaseTMA):
    nvtx_label = "optimized_threshold_tma"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        self.extension.threshold_tma_optimized(self.inputs, self.outputs, self.threshold)



def get_benchmark() -> ThresholdBenchmarkBaseTMA:
    return OptimizedThresholdTMABenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
