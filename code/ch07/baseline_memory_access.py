"""Ch7 baseline memory access benchmark (uncoalesced)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineMemoryAccessBenchmark(CudaBinaryBenchmark):
    """Wraps the uncoalesced CUDA kernel baseline."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_memory_access",
            friendly_name="Ch7 Baseline Memory Access",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics for memory_access."""
        from core.benchmark.metrics import compute_memory_access_metrics
        return compute_memory_access_metrics(
            bytes_requested=self._bytes_requested,
            bytes_actually_transferred=self._bytes_requested,  # Ideal case
            num_transactions=max(1, self._bytes_requested // 128),
            optimal_transactions=max(1, self._bytes_requested // 128),
        )


def get_benchmark() -> BaselineMemoryAccessBenchmark:
    return BaselineMemoryAccessBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nCh7 Baseline Memory Access: "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
