"""Optimized cluster-group benchmark that requires DSMEM + cluster launch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark
from common.python.hardware_capabilities import ensure_dsmem_supported

from ch10.cluster_group_utils import should_skip_cluster_error, raise_cluster_skip


class OptimizedClusterGroupBenchmark(CudaBinaryBenchmark):
    """Runs the DSMEM-enabled optimized kernel and fails fast when unsupported."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cluster_group",
            friendly_name="Optimized Cluster Group (DSMEM)",
            iterations=3,
            warmup=1,
            timeout_seconds=180,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def _ensure_cluster_support(self) -> None:
        try:
            ensure_dsmem_supported(description="Thread block cluster DSMEM benchmark")
        except RuntimeError as exc:
            raise_cluster_skip(str(exc))

    def setup(self) -> None:
        try:
            self._ensure_cluster_support()
            super().setup()
        except RuntimeError as exc:
            if should_skip_cluster_error(str(exc)):
                raise_cluster_skip(str(exc))
            raise

    def benchmark_fn(self) -> None:
        try:
            super().benchmark_fn()
        except RuntimeError as exc:
            if should_skip_cluster_error(str(exc)):
                raise_cluster_skip(str(exc))
            raise


def get_benchmark() -> OptimizedClusterGroupBenchmark:
    return OptimizedClusterGroupBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nOptimized Cluster Group (DSMEM): "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
