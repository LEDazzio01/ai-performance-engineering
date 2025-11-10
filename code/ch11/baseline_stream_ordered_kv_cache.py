"""baseline_stream_ordered_kv_cache.py - Single-stream KV cache updates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class BaselineStreamOrderedKvCacheBenchmark(Benchmark):
    """Baseline: update buffers sequentially on the default stream."""

    def __init__(self):
        self.device = resolve_device()
        self.data1: Optional[torch.Tensor] = None
        self.data2: Optional[torch.Tensor] = None
        self.data3: Optional[torch.Tensor] = None
        self.N = 5_000_000

    def setup(self) -> None:
        torch.manual_seed(42)
        self.data1 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.data2 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.data3 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.data1 is not None and self.data2 is not None and self.data3 is not None
        with nvtx_range("stream_ordered_kv_cache", enable=enable_nvtx):
            self.data1 = self.data1 * 2.0
            self.data2 = self.data2 * 2.0
            self.data3 = self.data3 * 2.0
            torch.cuda.synchronize()

    def teardown(self) -> None:
        self.data1 = None
        self.data2 = None
        self.data3 = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.data1 is None or self.data2 is None or self.data3 is None:
            return "Buffers not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineStreamOrderedKvCacheBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Stream-Ordered KV Cache: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
