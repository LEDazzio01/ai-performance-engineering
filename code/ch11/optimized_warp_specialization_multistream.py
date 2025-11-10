"""Optimized warp-specialization stream workload with overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedWarpSpecializationStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("warp_specialization_multistream")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedWarpSpecializationStreamsBenchmark()
