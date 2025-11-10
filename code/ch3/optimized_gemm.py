"""Optimized GEMM that runs a fused tensor-core matmul in one launch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn.functional as F

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class OptimizedGemmBenchmark(Benchmark):
    """Single large matmul captured inside torch.compile."""

    def __init__(self):
        self.device = resolve_device()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.epilogue: Optional[torch.Tensor] = None
        self.fn = None

    def setup(self) -> None:
        torch.manual_seed(1)
        enable_tf32()
        self.left = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.right = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.epilogue = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)

        def fused(a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            prod = torch.matmul(a, b)
            return F.gelu(prod + residual)

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            try:
                self.fn = compile_fn(fused, mode="reduce-overhead")
            except Exception:
                self.fn = fused
        else:
            self.fn = fused

        with torch.cuda.amp.autocast(dtype=torch.float16):
            for _ in range(3):
                _ = self.fn(self.left, self.right, self.epilogue)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.left is not None and self.right is not None and self.epilogue is not None
        op = self.fn
        with nvtx_range("optimized_gemm", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = op(self.left, self.right, self.epilogue)

    def teardown(self) -> None:
        self.left = None
        self.right = None
        self.epilogue = None
        self.fn = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=4)

    def validate_result(self) -> Optional[str]:
        if self.fn is None:
            return "Fused function not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedGemmBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized GEMM latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
