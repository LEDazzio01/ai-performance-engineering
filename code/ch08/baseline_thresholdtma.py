"""Baseline threshold benchmark gated for Blackwell TMA comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.threshold_tma_benchmark_base import ThresholdBenchmarkBaseTMA


class BaselineThresholdTMABenchmark(ThresholdBenchmarkBaseTMA):
    """Runs the branchy baseline but only on Blackwell/GB-series GPUs."""

    nvtx_label = "baseline_threshold_tma"
    _chunk_slices: list[tuple[int, int]]
    _chunk_cursor: int

    def __init__(self) -> None:
        super().__init__()
        self._chunk_slices = []
        self._chunk_cursor = 0

    def setup(self) -> None:
        super().setup()
        if self.host_inputs is None:
            return
        total = int(self.host_inputs.numel())
        chunk = max(1 << 21, total // 8)
        self._chunk_slices = [
            (offset, min(offset + chunk, total))
            for offset in range(0, total, chunk)
        ]

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        if self.host_inputs is not None:
            if not self._chunk_slices:
                self.inputs.copy_(self.host_inputs, non_blocking=False)
            else:
                total_chunks = len(self._chunk_slices)
                start_chunk = self._chunk_cursor % total_chunks
                for step in range(total_chunks):
                    chunk_idx = (start_chunk + step) % total_chunks
                    start, end = self._chunk_slices[chunk_idx]
                    self.inputs[start:end].copy_(self.host_inputs[start:end], non_blocking=False)
                self._chunk_cursor += 1
        self.extension.threshold_tma_baseline(self.inputs, self.outputs, self.threshold)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for thresholdtma."""
        from core.benchmark.metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="thresholdtma",
        )

def get_benchmark() -> ThresholdBenchmarkBaseTMA:
    return BaselineThresholdTMABenchmark()


def main() -> None:
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = BaselineThresholdTMABenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Baseline Threshold (Blackwell TMA gate)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
