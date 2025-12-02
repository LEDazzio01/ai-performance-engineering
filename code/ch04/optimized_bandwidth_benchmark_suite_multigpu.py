"""Optimized bandwidth benchmark suite; runs P2P copies across GPUs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from ch04.baseline_bandwidth_benchmark_suite_multigpu import measure_peer_bandwidth
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedBandwidthSuiteMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
        self.last_bandwidth_gbps: Optional[float] = None

    def benchmark_fn(self) -> None:
        self.last_bandwidth_gbps = measure_peer_bandwidth(size_mb=512, iterations=75, async_copy=True)

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=30)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return measured P2P bandwidth."""
        return {"p2p_bandwidth_gbps": float(self.last_bandwidth_gbps or 0.0)}

def get_benchmark() -> BaseBenchmark:
    return OptimizedBandwidthSuiteMultiGPU()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
