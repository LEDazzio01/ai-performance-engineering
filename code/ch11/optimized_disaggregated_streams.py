"""Optimized disaggregated stream workload with overlapping streams."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedDisaggregatedStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("disaggregated_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedDisaggregatedStreamsBenchmark()
