"""Python wrapper for baseline_cluster_group_no_dsmem.cu."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark

from ch10.cluster_group_utils import raise_cluster_skip


class BaselineClusterGroupNoDSMEMBenchmark(CudaBinaryBenchmark):
    """Wraps the DSMEM-free baseline cluster reduction."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cluster_group_no_dsmem",
            friendly_name="Baseline Cluster Group (no DSMEM)",
            iterations=3,
            warmup=1,
            timeout_seconds=180,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )

    def benchmark_fn(self) -> None:
        try:
            super().benchmark_fn()
        except RuntimeError as exc:
            raise_cluster_skip(str(exc))


def get_benchmark() -> BaselineClusterGroupNoDSMEMBenchmark:
    return BaselineClusterGroupNoDSMEMBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Cluster Group (no DSMEM): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
