"""Optimized GEMM stream workload with overlapping streams."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedGemmStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("gemm_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedGemmStreamsBenchmark()
