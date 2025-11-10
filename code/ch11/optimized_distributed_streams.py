"""Optimized distributed-style overlap using multiple CUDA streams."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedDistributedStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("distributed_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedDistributedStreamsBenchmark()
