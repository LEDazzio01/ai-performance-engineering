"""stream_ordered_kv_cache.py - Concurrent kernel execution with streams (optimized)."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedStreamOrderedKvCacheBenchmark(ConcurrentStreamOptimized):
    """Optimized: overlap KV cache updates across CUDA streams."""

    def __init__(self):
        super().__init__(
            "stream_ordered_kv_cache",
            num_elements=18_000_000,
            num_segments=8,
            num_streams=2,
        )


def get_benchmark() -> OptimizedStreamOrderedKvCacheBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStreamOrderedKvCacheBenchmark()
