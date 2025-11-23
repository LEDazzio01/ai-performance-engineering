"""Optimized NVSHMEM training example with NCCL 2.28 tuning; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from ch4.nccl_blackwell_config import (
    configure_nccl_for_8xB200,
    configure_nccl_for_blackwell,
    configure_nccl_for_gb200_gb300,
    detect_8xb200_topology,
)
from ch4.nvshmem_training_example import main as nvshmem_train_main
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


def _configure_blackwell_nccl() -> None:
    """Enable NCCL 2.28 knobs for Blackwell/Grace-Blackwell."""
    try:
        topo = detect_8xb200_topology()
    except Exception:
        configure_nccl_for_blackwell(verbose=False)
        return

    if topo.get("has_grace_cpu"):
        configure_nccl_for_gb200_gb300(verbose=False)
    elif topo.get("num_gpus", 0) >= 8 and topo.get("is_8xb200"):
        configure_nccl_for_8xB200(verbose=False)
    else:
        configure_nccl_for_blackwell(verbose=False)


class OptimizedNVSHMEMTrainingExampleMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_training_example requires >=2 GPUs")
        _configure_blackwell_nccl()

    def benchmark_fn(self) -> None:
        # Prefer the pipeline demo to exercise NVLink5/NVLink-C2C fast paths.
        original_argv = sys.argv[:]
        try:
            sys.argv = [original_argv[0], "--demo", "pipeline"]
            nvshmem_train_main()
        finally:
            sys.argv = original_argv

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=1, measurement_timeout_seconds=300)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMTrainingExampleMultiGPU()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"NVSHMEM training example optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
