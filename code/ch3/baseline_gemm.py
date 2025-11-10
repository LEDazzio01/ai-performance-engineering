"""Baseline GEMM that serializes micro-batches with CPU synchronization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class BaselineGemmBenchmark(Benchmark):
    """Splits a large GEMM into many small kernels with extra CPU sync."""

    def __init__(self):
        self.device = resolve_device()
        self.block = 512
        self.blocks = 8
        self.left_blocks: List[torch.Tensor] = []
        self.right_blocks: List[torch.Tensor] = []

    def setup(self) -> None:
        torch.manual_seed(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.left_blocks = [
            torch.randn(self.block, self.block, device=self.device, dtype=torch.float32)
            for _ in range(self.blocks)
        ]
        self.right_blocks = [
            torch.randn(self.block, self.block, device=self.device, dtype=torch.float32)
            for _ in range(self.blocks)
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        total = torch.zeros(self.block, self.block, device=self.device, dtype=torch.float32)
        with nvtx_range("baseline_gemm", enable=enable_nvtx):
            for a, b in zip(self.left_blocks, self.right_blocks):
                total += torch.matmul(a, b)
                torch.cuda.synchronize()  # simulate CPU scheduling between launches
        return total

    def teardown(self) -> None:
        self.left_blocks = []
        self.right_blocks = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=4)

    def validate_result(self) -> Optional[str]:
        if not self.left_blocks or not self.right_blocks:
            return "Blocks not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineGemmBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline GEMM latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
