"""Optimized stream workload with tiled, coalesced overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedTilingStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("tiling_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedTilingStreamsBenchmark()
