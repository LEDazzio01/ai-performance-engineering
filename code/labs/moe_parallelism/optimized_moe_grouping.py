"""Optimized expert placement: EP per node with router slack."""

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
        name="Optimized expert grouping (EP kept on-node)",
        dp=4,
        pp=4,
        tp=2,
        ep=4,
        microbatch_sequences=32,
        microbatches=18,
        experts_per_gpu=4,
        capacity_factor=1.25,
        dense_checkpoint_fraction=0.5,
        moe_checkpoint_fraction=1.0,
        stage_layers=[24, 24, 24, 24],
        cross_node_ep=False,
        notes=[
            "EP groups fit inside a single node so MoE all-to-all stays on NVSwitch",
            "Each GPU hosts four experts (32 per stage) which aligns with the 128-expert target",
            "Router slack (capacity 1.25) + top-2 gating keeps load balance auxiliary loss effective",
        ],
    )


class OptimizedMoeGroupingBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return OptimizedMoeGroupingBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
