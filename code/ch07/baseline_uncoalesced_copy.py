"""Python harness wrapper for ch07's baseline_copy_uncoalesced.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCopyUncoalescedBenchmark(CudaBinaryBenchmark):
    """Wraps the strided copy baseline kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 23
        repeat = 40
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_copy_uncoalesced",
            friendly_name="Ch7 Uncoalesced Copy",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "N": n_elems,
                "repeat": repeat,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * 8),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> BaselineCopyUncoalescedBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCopyUncoalescedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
