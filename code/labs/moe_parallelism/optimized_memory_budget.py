"""Optimized memory sizing: checkpoint-aware micro-batch and TP."""

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
        name="Optimized memory budget (balanced activations)",
        dp=4,
        pp=4,
        tp=2,
        ep=4,
        microbatch_sequences=32,
        microbatches=16,
        experts_per_gpu=4,
        capacity_factor=1.25,
        dense_checkpoint_fraction=0.5,
        moe_checkpoint_fraction=0.85,
        stage_layers=[24, 24, 24, 24],
        cross_node_ep=False,
        notes=[
            "Tensor parallelism halves the hidden slice per GPU, freeing activation headroom",
            "Checkpoint every other dense block + 85% of MoE blocks to keep margin ≥15 GB",
            "Micro-batch capped at 32 sequences so we can still hold 16 chunks in flight",
        ],
    )


class OptimizedMemoryBudgetBenchmark(PlanBenchmark):
    def __init__(self) -> None:
        super().__init__(build_plan())

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return OptimizedMemoryBudgetBenchmark()


if __name__ == "__main__":
    run_benchmark(get_benchmark())
