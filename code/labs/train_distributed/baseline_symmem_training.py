"""Baseline symmetric memory training demo (fallback path, symmetric disabled)."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parents[2] / "ch04" / "symmetric_memory_training_advanced.py",
        base_args=["--demo", "optimizer", "--disable-symmetric"],
        target_label="labs/train_distributed:symmem_training",
        multi_gpu_required=True,
        name="baseline_symmem_training",
    )
