"""Benchmark wrapper for bandwidth suite; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

import os
import time
from typing import Optional


def measure_peer_bandwidth(size_mb: int = 256, iterations: int = 50, async_copy: bool = False) -> float:
    """
    Measure GPU-to-GPU bandwidth by copying a large tensor between two devices.
    """
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
    bytes_per_iter = size_mb * 1024 * 1024
    numel = bytes_per_iter // 4  # float32
    src = torch.randn(numel, device="cuda:0")
    dst = torch.empty_like(src, device="cuda:1")
    stream = torch.cuda.Stream(device="cuda:1") if async_copy else None

    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")
    start = time.perf_counter()
    for _ in range(iterations):
        if stream is not None:
            with torch.cuda.stream(stream):
                dst.copy_(src, non_blocking=True)
        else:
            dst.copy_(src, non_blocking=False)
    if stream is not None:
        stream.synchronize()
    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")
    elapsed = (time.perf_counter() - start) / iterations
    gb_per_iter = bytes_per_iter / 1e9
    return gb_per_iter / elapsed


class BandwidthSuiteMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
        self.last_bandwidth_gbps: Optional[float] = None

    def benchmark_fn(self) -> None:
        self.last_bandwidth_gbps = measure_peer_bandwidth()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=30,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return measured P2P bandwidth."""
        return {"p2p_bandwidth_gbps": float(self.last_bandwidth_gbps or 0.0)}

def get_benchmark() -> BaseBenchmark:
    return BandwidthSuiteMultiGPU()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
