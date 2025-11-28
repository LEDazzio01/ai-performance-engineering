"""Benchmark wrapper for bandwidth suite; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig

import os
import time
def run_quick_bandwidth_smoke() -> float:
    """Ultra-light smoke: single GPU tensor add and sync."""
    device = torch.device("cuda:0")
    a = torch.randn(1024, device=device)
    b = torch.randn(1024, device=device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        _ = a + b
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 5
    return elapsed * 1000.0  # ms


class BandwidthSuiteMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("SKIPPED: bandwidth smoke runs single-process; launch with --launch-via python")

    def benchmark_fn(self) -> None:
        run_quick_bandwidth_smoke()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=30,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

def get_benchmark() -> BaseBenchmark:
    return BandwidthSuiteMultiGPU()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"Bandwidth suite (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
