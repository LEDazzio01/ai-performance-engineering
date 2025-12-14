"""Branch-heavy threshold benchmark baseline."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.threshold_benchmark_base import ThresholdBenchmarkBase


class BaselineThresholdBenchmark(ThresholdBenchmarkBase):
    nvtx_label = "baseline_threshold"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        # Naive kernel with branch divergence - data is already on GPU
        # The kernel uses nested if/else branches that cause warp divergence
        self.extension.threshold_baseline(self.inputs, self.outputs, self.threshold)



def get_benchmark() -> ThresholdBenchmarkBase:
    return BaselineThresholdBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
