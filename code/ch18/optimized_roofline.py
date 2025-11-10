"""Roofline-aware implementation that maximizes arithmetic intensity."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch18.workload_config import WORKLOAD, is_smoke_test

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedRooflineBenchmark(Benchmark):
    """Fuses the entire matmul and computes intensity statistics."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.matrix_size = self.workload.roofline_matmul_size
        self.activation: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.activation = torch.randn(
            self.matrix_size,
            self.matrix_size,
            dtype=torch.float16,
            device=self.device,
        )
        self.weights = torch.randn(
            self.matrix_size,
            self.matrix_size,
            dtype=torch.float16,
            device=self.device,
        )
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.activation is not None
        assert self.weights is not None

        with nvtx_range("optimized_roofline", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = torch.matmul(self.activation, self.weights)
            bytes_accessed = (self.activation.numel() + self.weights.numel() + output.numel()) * output.element_size()
            flops = 2 * (self.matrix_size ** 3)
            intensity = flops / bytes_accessed
            if intensity <= 0:
                raise RuntimeError("Invalid roofline intensity")

    def teardown(self) -> None:
        self.activation = None
        self.weights = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=2, warmup=1, enable_memory_tracking=False)

    def validate_result(self) -> Optional[str]:
        if self.activation is None or self.weights is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedRooflineBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=2, warmup=1))
    result = harness.benchmark(OptimizedRooflineBenchmark())
    print(f"Optimized roofline mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
