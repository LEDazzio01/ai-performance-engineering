"""Baseline network layout: DP-heavy plan with no affinity awareness."""

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
        name="Baseline network affinity (DP-dominated)",
        dp=8,
        pp=2,
        tp=1,
        ep=8,
        microbatch_sequences=16,
        microbatches=10,
        experts_per_gpu=4,
        capacity_factor=1.1,
        dense_checkpoint_fraction=0.75,
        moe_checkpoint_fraction=1.0,
        stage_layers=[48, 48],
        cross_node_ep=False,
        notes=[
            "Eight DP replicas cause full-parameter all-reduce over HDR100 every step",
            "Only two pipeline stages so activation transfers are massive and cross-node",
            "No affinity guidance for NIC binding or NVSwitch locality",
        ],
    )


class BaselineNetworkAffinityBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return BaselineNetworkAffinityBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
