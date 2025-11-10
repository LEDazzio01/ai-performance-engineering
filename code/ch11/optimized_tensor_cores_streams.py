"""Optimized tensor-core stream workload with overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedTensorCoresStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("tensor_cores_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedTensorCoresStreamsBenchmark()
