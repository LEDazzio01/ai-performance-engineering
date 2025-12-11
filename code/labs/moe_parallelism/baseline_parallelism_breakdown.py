"""Baseline: inconsistent DP×PP×TP×EP layout for the MoE lab."""

from __future__ import annotations


import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.moe_parallelism.plan import ParallelismPlan
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark


def build_plan() -> ParallelismPlan:
    return ParallelismPlan(
        name="Baseline parallelism factorization (underspecified)",
        dp=2,
        pp=16,
        tp=1,
        ep=2,
        microbatch_sequences=16,
        microbatches=8,
        experts_per_gpu=8,
        capacity_factor=1.0,
        dense_checkpoint_fraction=1.0,
        moe_checkpoint_fraction=1.0,
        stage_layers=[6] * 16,
        cross_node_ep=True,
        notes=[
            "Intentionally mismatched world size (only 64 ranks assigned to a 128 GPU cluster)",
            "Expert groups bleed across nodes so token exchanges pound HDR100 instead of NVSwitch",
            "TP degree of 1 leaves each GPU with the full hidden size, inflating activation memory",
        ],
    )


class BaselineParallelismBreakdownBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return BaselineParallelismBreakdownBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
