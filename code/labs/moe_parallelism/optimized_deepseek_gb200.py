"""Optimized plan for DeepSeek-R1-678B on GB200 NVL72 (8 racks, IB)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.moe_parallelism.plan import ParallelismPlan, SPEC_PRESETS
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark


CLUSTER, MODEL = SPEC_PRESETS["deepseek_r1_678b_gb200_ib"]


def build_plan() -> ParallelismPlan:
    return ParallelismPlan(
        name="Optimized DeepSeek-R1-678B (DP8×PP6×TP3×EP4)",
        dp=8,
        pp=6,
        tp=3,
        ep=4,
        microbatch_sequences=32,
        microbatches=36,
        experts_per_gpu=4,
        capacity_factor=1.2,
        dense_checkpoint_fraction=0.5,
        moe_checkpoint_fraction=0.9,
        stage_layers=[22, 21, 21, 21, 21, 22],
        cross_node_ep=False,
        notes=[
            "PP=6 with 36 micro-batches reduces bubble vs the 8-stage baseline",
            "TP=3 lowers per-rank params/activations to ease HBM pressure",
            "Stages rebalanced to keep early/late blocks heavier for embeds/head",
        ],
    )


class OptimizedDeepseekGb200Benchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(plan=build_plan(), cluster=CLUSTER, model=MODEL)

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return OptimizedDeepseekGb200Benchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
