"""Optimized GEMM that mirrors CUTLASS-style fused tensor-core kernels."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class OptimizedCutlassBenchmark(Benchmark):
    """Uses FP16 inputs, TF32 accumulation, and torch.compile fusion."""

    def __init__(self):
        self.device = resolve_device()
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        self.compiled_op = None
        self.m = 3072
        self.n = 3072
        self.k = 3072

    def setup(self) -> None:
        torch.manual_seed(42)
        enable_tf32()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.bias = torch.randn(self.n, device=self.device, dtype=torch.float16)

        def fused_op(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            prod = torch.matmul(a, b)
            return torch.nn.functional.gelu(prod + bias)

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            try:
                self.compiled_op = compile_fn(fused_op, mode="reduce-overhead")
            except Exception:
                self.compiled_op = fused_op
        else:
            self.compiled_op = fused_op

        # Warm up Lt heuristics so later iterations stay steady.
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for _ in range(5):
                _ = self.compiled_op(self.A, self.B, self.bias)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.A is not None and self.B is not None and self.bias is not None
        op = self.compiled_op
        with nvtx_range("optimized_cutlass", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = op(self.A, self.B, self.bias)

    def teardown(self) -> None:
        self.A = None
        self.B = None
        self.bias = None
        self.compiled_op = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=4)

    def validate_result(self) -> Optional[str]:
        if self.compiled_op is None:
            return "Compiled op unavailable"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedCutlassBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized CUTLASS latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
