"""Baseline memory sizing: oversized micro-batch + no checkpointing."""

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
        name="Baseline memory budget (overcommitted HBM)",
        dp=4,
        pp=4,
        tp=1,
        ep=8,
        microbatch_sequences=64,
        microbatches=24,
        experts_per_gpu=4,
        capacity_factor=1.1,
        dense_checkpoint_fraction=1.0,
        moe_checkpoint_fraction=1.0,
        stage_layers=[24, 24, 24, 24],
        cross_node_ep=False,
        notes=[
            "Full hidden state per GPU plus 64-seq micro-batch blows past 80 GB",
            "No activation checkpointing, so dense blocks stash every tensor",
            "24 in-flight micro-batches also inflate optimizer/shard residency",
        ],
    )


class BaselineMemoryBudgetBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return BaselineMemoryBudgetBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
