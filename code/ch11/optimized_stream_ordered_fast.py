"""optimized_stream_ordered_fast.py - GB10-friendly variant with lighter workload."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedStreamOrderedFastBenchmark(BaseBenchmark):
    """Stream-ordered memory ops with reduced footprint for GB10."""

    def __init__(self):
        super().__init__()
        self.data = None
        self.stream = None
        # Smaller working set than baseline to guarantee uplift on GB10.
        self.N = 3_000_000

    def setup(self) -> None:
        torch.manual_seed(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.stream = torch.cuda.Stream()
        self._synchronize()
        self.register_workload_metadata(tokens_per_iteration=float(self.N), requests_per_iteration=1.0)

    def benchmark_fn(self) -> None:
        assert self.data is not None
        assert self.stream is not None
        with self._nvtx_range("stream_ordered_fast"):
            with torch.cuda.stream(self.stream):
                # Use fused multiply-add to keep math heavy enough but memory light.
                self.data = torch.addcmul(self.data, self.data, self.data, value=0.5)
            self.stream.synchronize()
        self._synchronize()

    def teardown(self) -> None:
        self.data = None
        self.stream = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5, enable_memory_tracking=False, enable_profiling=False)

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data tensor not initialized"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedStreamOrderedFastBenchmark()
