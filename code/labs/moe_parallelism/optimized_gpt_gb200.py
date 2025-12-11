"""Optimized plan for GPT-OSS-120B on GB200 NVL72 (8 racks, IB)."""

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
        name="Optimized GPT-OSS-120B (DP8×PP6×TP3×EP4)",
        dp=8,
        pp=6,
        tp=3,
        ep=4,
        microbatch_sequences=64,
        microbatches=32,
        experts_per_gpu=4,
        capacity_factor=1.2,
        dense_checkpoint_fraction=0.5,
        moe_checkpoint_fraction=0.85,
        stage_layers=[18, 16, 16, 16, 15, 15],
        cross_node_ep=False,
        notes=[
            "PP reduced to 6 with deeper micro-batching (32) to cut bubble",
            "Stage0 takes extra layers for embeddings/head; later stages balanced",
            "TP raised to 3 to trim per-rank params/activations without leaving NVLink",
        ],
    )


class OptimizedGptGb200Benchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(plan=build_plan(), cluster=CLUSTER, model=MODEL)

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return OptimizedGptGb200Benchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
