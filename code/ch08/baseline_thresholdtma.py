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



def get_benchmark() -> ThresholdBenchmarkBaseTMA:
    return BaselineThresholdTMABenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
