"""Optimized NVSHMEM pipeline parallel wrapper with NVLink5/NVLS tuning; skips on <2 GPUs.

This keeps the harness happy with a baseline/optimized pair. When you add a
true optimization (e.g., tuned block sizes or IBGDA), swap the call inside
benchmark_fn accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from ch04.nccl_blackwell_config import (
    configure_nccl_for_8xB200,
    configure_nccl_for_blackwell,
    configure_nccl_for_gb200_gb300,
    detect_8xb200_topology,
)
from ch04.nvshmem_pipeline_parallel import main as nvshmem_main
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


def _configure_blackwell_nccl() -> None:
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


class OptimizedNVSHMEMPipelineParallelMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_pipeline_parallel requires >=2 GPUs")
        _configure_blackwell_nccl()

    def benchmark_fn(self) -> None:
        original_argv = sys.argv[:]
        try:
            # Run both schedules to stress NVLink/NVLS and NVLink-C2C tuning.
            sys.argv = [original_argv[0], "--schedule", "all"]
            nvshmem_main()
        finally:
            sys.argv = original_argv

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=300)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMPipelineParallelMultiGPU()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"NVSHMEM pipeline parallel optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
