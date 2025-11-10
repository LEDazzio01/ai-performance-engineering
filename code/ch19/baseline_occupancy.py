"""baseline_occupancy.py - Low occupancy baseline using CUDA kernels."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from common.python.occupancy import run_low_occupancy


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class BaselineOccupancyBenchmark(Benchmark):
    """Low-occupancy CUDA kernel (small blocks, few resident warps)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.work_iters = 2048
        self.N = 16_777_216  # 16M elements (~64MB)
        self.input = None
        self.output = None
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("occupancy", enable=enable_nvtx):
            run_low_occupancy(self.input, self.output, self.work_iters)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        if self.input is None or self.output is None:
            return "Input/output tensors not initialized"
        if self.output.shape != self.input.shape:
            return "Shape mismatch between input and output"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    return BaselineOccupancyBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Occupancy (Low): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
