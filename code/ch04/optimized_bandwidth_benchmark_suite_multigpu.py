"""Optimized (placeholder) bandwidth benchmark suite; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from ch04.baseline_bandwidth_benchmark_suite_multigpu import run_quick_bandwidth_smoke
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedBandwidthSuiteMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("SKIPPED: bandwidth smoke runs single-process; launch with --launch-via python")

    def benchmark_fn(self) -> None:
        # Placeholder optimized path: reuse smoke test for now.
        run_quick_bandwidth_smoke()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=30)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedBandwidthSuiteMultiGPU()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    print(f"Bandwidth suite optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
