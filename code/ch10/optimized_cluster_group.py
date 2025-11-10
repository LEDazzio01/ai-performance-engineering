"""Python harness wrapper for optimized cluster group binaries."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark

from ch10.cluster_group_utils import raise_cluster_skip, should_skip_cluster_error


class OptimizedClusterGroupBenchmark(CudaBinaryBenchmark):
    """Simple passthrough used for typing."""


class ClusterGroupWithFallback(OptimizedClusterGroupBenchmark):
    """Try multiple binaries so we keep working when DSMEM isn't available."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        variants = [
            "optimized_cluster_group",
            "optimized_cluster_group_single_cta",
            "optimized_cluster_group_no_dsmem",
        ]
        self._candidates = [
            CudaBinaryBenchmark(
                chapter_dir=chapter_dir,
                binary_name=name,
                friendly_name=f"Optimized Cluster Group ({name})",
                iterations=3,
                warmup=1,
                timeout_seconds=180,
                time_regex=r"TIME_MS:\s*([0-9.]+)",
            )
            for name in variants
        ]
        self._active: CudaBinaryBenchmark | None = None
        self._candidate_index = -1

    def setup(self) -> None:
        last_error: RuntimeError | None = None
        for idx, candidate in enumerate(self._candidates):
            try:
                candidate.setup()
                self._active = candidate
                self._candidate_index = idx
                return
            except RuntimeError as exc:
                if should_skip_cluster_error(str(exc)):
                    last_error = exc
                    continue
                raise
        raise_cluster_skip(str(last_error) if last_error else "cluster launch unavailable")

    def benchmark_fn(self) -> None:
        if self._active is None:
            raise RuntimeError("Cluster benchmark not initialized")
        try:
            self._active.benchmark_fn()
        except RuntimeError as exc:
            if not should_skip_cluster_error(str(exc)):
                raise
            for next_idx in range(self._candidate_index + 1, len(self._candidates)):
                candidate = self._candidates[next_idx]
                try:
                    candidate.setup()
                    self._active = candidate
                    self._candidate_index = next_idx
                    candidate.benchmark_fn()
                    return
                except RuntimeError as inner_exc:
                    if should_skip_cluster_error(str(inner_exc)):
                        continue
                    raise
            raise_cluster_skip(str(exc))

    def teardown(self) -> None:
        # Nothing to clean up; subprocesses handle their own lifetime.
        pass

    def get_config(self):
        if self._active is not None:
            return self._active.get_config()
        return self._candidates[0].get_config()

    def validate_result(self):
        if self._active is None:
            return "Cluster benchmark not initialized"
        return self._active.validate_result()


def get_benchmark() -> ClusterGroupWithFallback:
    return ClusterGroupWithFallback()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Cluster Group: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
