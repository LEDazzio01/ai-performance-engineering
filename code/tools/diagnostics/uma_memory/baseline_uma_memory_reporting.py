"""Baseline UMA memory reporting using cudaMemGetInfo only."""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Optional

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from ch2.uma_memory_utils import format_bytes, is_integrated_gpu, read_meminfo


class BaselineUmaMemoryReportingBenchmark(BaseBenchmark):
    """Relies solely on cudaMemGetInfo with no UMA adjustment."""

    def __init__(self):
        super().__init__()
        self.cuda_free_bytes = 0
        self.cuda_total_bytes = 0
        self.host_available_bytes: Optional[int] = None
        self.swap_free_bytes: Optional[int] = None

    def setup(self) -> None:
        torch.cuda.empty_cache()
        self._sample()

    def _sample(self) -> None:
        free, total = torch.cuda.mem_get_info()
        self.cuda_free_bytes = free
        self.cuda_total_bytes = total
        snapshot = read_meminfo()
        if snapshot:
            self.host_available_bytes = snapshot.effective_available_kb() * 1024
            self.swap_free_bytes = snapshot.swap_free_kb * 1024
        else:
            self.host_available_bytes = None
            self.swap_free_bytes = None

    def benchmark_fn(self) -> None:
        self._sample()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            enable_memory_tracking=False,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        metrics: Dict[str, float] = {
            "cuda_free_gb": self.cuda_free_bytes / (1024**3),
            "cuda_total_gb": self.cuda_total_bytes / (1024**3),
        }
        if self.host_available_bytes is not None:
            metrics["host_memavailable_gb"] = self.host_available_bytes / (1024**3)
        if self.swap_free_bytes is not None:
            metrics["swap_free_gb"] = self.swap_free_bytes / (1024**3)
        return metrics


def summarize() -> None:
    bench = BaselineUmaMemoryReportingBenchmark()
    bench.setup()
    bench.benchmark_fn()
    integrated = is_integrated_gpu()
    print("\n=== Baseline CUDA memory report ===")
    print(f"Integrated GPU detected: {integrated}")
    print(f"cudaMemGetInfo free:  {format_bytes(bench.cuda_free_bytes)}")
    print(f"cudaMemGetInfo total: {format_bytes(bench.cuda_total_bytes)}")
    if bench.host_available_bytes is not None:
        print(f"Host MemAvailable:   {format_bytes(bench.host_available_bytes)}")
    if bench.swap_free_bytes is not None:
        print(f"SwapFree:            {format_bytes(bench.swap_free_bytes)}")


def get_benchmark() -> BaseBenchmark:
    return BaselineUmaMemoryReportingBenchmark()


if __name__ == "__main__":
    summarize()
