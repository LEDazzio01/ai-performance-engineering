"""Roofline baseline that uses tiny tiles and forces low arithmetic intensity."""

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


class BaselineRooflineBenchmark(Benchmark):
    """Breaks matmul into tiny tiles, thrashing memory bandwidth."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.matrix_size = self.workload.roofline_matmul_size
        self.tile = self.workload.roofline_tile
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
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.activation is not None
        assert self.weights is not None

        with nvtx_range("baseline_roofline", enable=enable_nvtx):
            result = 0.0
            for row in range(0, self.matrix_size, self.tile):
                a_block = self.activation[row : row + self.tile, :].contiguous().float()
                for col in range(0, self.matrix_size, self.tile):
                    b_block = self.weights[:, col : col + self.tile].contiguous().float()
                    partial = torch.matmul(a_block, b_block)
                    result += partial.sum().item()
                    torch.cuda.synchronize()
            if result == float("inf"):
                raise RuntimeError("Prevent compiler from removing loop")

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
    return BaselineRooflineBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=2, warmup=1))
    result = harness.benchmark(BaselineRooflineBenchmark())
    print(f"Baseline roofline mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
