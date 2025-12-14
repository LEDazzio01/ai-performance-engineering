"""Baseline AI optimization benchmark with low ILP."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.ai_optimization_benchmark_base import AiOptimizationBenchmarkBase


class BaselineAiOptimizationBenchmark(AiOptimizationBenchmarkBase):
    nvtx_label = "baseline_ai_optimization"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.weights is not None
        assert self.output is not None
        self.extension.ai_baseline(self.inputs, self.weights, self.output)



def get_benchmark() -> AiOptimizationBenchmarkBase:
    return BaselineAiOptimizationBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
