"""Baseline expert placement: cross-node EP groups with weak capacity settings."""

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
        name="Baseline expert grouping (cross-node EP=8)",
        dp=2,
        pp=4,
        tp=2,
        ep=8,
        microbatch_sequences=24,
        microbatches=12,
        experts_per_gpu=2,
        capacity_factor=0.9,
        dense_checkpoint_fraction=0.75,
        moe_checkpoint_fraction=1.0,
        stage_layers=[24, 24, 24, 24],
        cross_node_ep=True,
        notes=[
            "Each pipeline stage spans two nodes so EP groups straddle HDR100 links",
            "Capacity factor 1.0 gives no safety margin for hot experts, so drops spike",
            "Per-GPU tokens stay high because micro-batch partitioning ignores EP fan-out",
        ],
    )


class BaselineMoeGroupingBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return BaselineMoeGroupingBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
