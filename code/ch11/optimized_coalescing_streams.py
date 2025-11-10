"""Optimized stream workload with coalesced, overlapping transfers."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedCoalescingStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("coalescing_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedCoalescingStreamsBenchmark()
