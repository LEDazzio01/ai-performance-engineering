"""Baseline plan for GPT-OSS-120B on GB200 NVL72 (8 racks, IB)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.moe_parallelism.plan import ParallelismPlan, SPEC_PRESETS
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark


CLUSTER, MODEL = SPEC_PRESETS["gpt_oss_120b_gb200_ib"]


def build_plan() -> ParallelismPlan:
    return ParallelismPlan(
        name="Baseline GPT-OSS-120B (DP9×PP8×TP2×EP4)",
        dp=9,
        pp=8,
        tp=2,
        ep=4,
        microbatch_sequences=48,
        microbatches=24,
        experts_per_gpu=4,
        capacity_factor=1.2,
        dense_checkpoint_fraction=0.6,
        moe_checkpoint_fraction=0.9,
        stage_layers=[12] * 8,
        cross_node_ep=False,
        notes=[
            "Matches the original GB200 NVL72 layout with PP=8 (bubble ~23%)",
            "Stage splits are even; no special handling for embedding/head",
        ],
    )


class BaselineGptGb200Benchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(plan=build_plan(), cluster=CLUSTER, model=MODEL)

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return BaselineGptGb200Benchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
