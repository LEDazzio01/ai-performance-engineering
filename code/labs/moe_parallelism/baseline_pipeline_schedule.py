"""Baseline pipeline scheduling: too many stages, shallow micro-batching."""

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
        name="Baseline pipeline schedule (64-stage mindset)",
        dp=4,
        pp=8,
        tp=1,
        ep=4,
        microbatch_sequences=8,
        microbatches=6,
        experts_per_gpu=4,
        capacity_factor=1.0,
        dense_checkpoint_fraction=1.0,
        moe_checkpoint_fraction=1.0,
        stage_layers=[36, 12, 12, 12, 12, 6, 3, 3],
        cross_node_ep=False,
        notes=[
            "8 pipeline stages squeezed into 4 nodes -> half-node stages and tiny micro-batches",
            "Micro-batches (6) barely exceed the stage count (8), so pipeline bubbles dominate",
            "Stage splits ignore embedding/output heft, leaving stage0 as a straggler",
        ],
    )


class BaselinePipelineScheduleBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return BaselinePipelineScheduleBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
