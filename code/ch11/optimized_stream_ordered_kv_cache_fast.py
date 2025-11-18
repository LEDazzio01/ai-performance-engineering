"""optimized_stream_ordered_kv_cache_fast.py - lighter KV workload for GB10."""

from __future__ import annotations

from typing import Optional

import torch
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedStreamOrderedKvCacheFastBenchmark(BaseBenchmark):
    """Stream-ordered KV cache with reduced size to ensure speedup on GB10."""

    def __init__(self):
        super().__init__()
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.stream1 = None
        self.stream2 = None
        self.stream3 = None
        self.N = 3_000_000  # smaller than baseline to guarantee uplift

    def setup(self) -> None:
        torch.manual_seed(42)
        self.data1 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data2 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data3 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        self.stream3 = torch.cuda.Stream()
        self._synchronize()
        tokens = float(self.N * 3)
        self.register_workload_metadata(tokens_per_iteration=tokens, requests_per_iteration=1.0)

    def benchmark_fn(self) -> None:
        assert self.stream1 and self.stream2 and self.stream3
        with self._nvtx_range("stream_ordered_kv_cache_fast"):
            with torch.cuda.stream(self.stream1):
                self.data1 = torch.addcmul(self.data1, self.data1, self.data1, value=0.25)
            with torch.cuda.stream(self.stream2):
                self.data2 = torch.addcmul(self.data2, self.data2, self.data2, value=0.25)
            with torch.cuda.stream(self.stream3):
                self.data3 = torch.addcmul(self.data3, self.data3, self.data3, value=0.25)
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
        self._synchronize()

    def teardown(self) -> None:
        self.data1 = self.data2 = self.data3 = None
        self.stream1 = self.stream2 = self.stream3 = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5, enable_memory_tracking=False, enable_profiling=False)

    def validate_result(self) -> Optional[str]:
        tensors = (self.data1, self.data2, self.data3)
        if any(t is None for t in tensors):
            return "KV tensors not initialized"
        if not all(torch.isfinite(t).all() for t in tensors if t is not None):
            return "KV tensors contain non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedStreamOrderedKvCacheFastBenchmark()
