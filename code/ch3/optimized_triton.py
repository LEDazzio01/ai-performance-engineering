"""Optimized kernel implemented with Triton to fuse sin/scale/bias."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from common.python import triton_compat  # noqa: F401

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3 triton example")
    return torch.device("cuda")


if TRITON_AVAILABLE:

    @triton.jit
    def fused_kernel(x_ptr, scale, bias, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale_val = tl.load(scale).to(tl.float32)
        bias_val = tl.load(bias).to(tl.float32)
        y = tl.sin(x)
        y = y * scale_val
        y = y + bias_val
        tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)


class OptimizedTritonBenchmark(Benchmark):
    """Fuses the entire elementwise op into one Triton kernel."""

    def __init__(self):
        self.device = resolve_device()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(55)
        self.input = torch.randn(8_388_608, device=self.device, dtype=torch.float16)
        self.output = torch.empty_like(self.input)
        self.scale = torch.tensor([1.7], device=self.device, dtype=torch.float16)
        self.bias = torch.tensor([-0.5], device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.input is not None and self.output is not None
        assert self.scale is not None and self.bias is not None
        with nvtx_range("optimized_triton", enable=enable_nvtx):
            if TRITON_AVAILABLE:
                grid = lambda meta: (triton.cdiv(self.input.numel(), meta["BLOCK_SIZE"]),)
                fused_kernel[grid](
                    self.input,
                    self.scale,
                    self.bias,
                    self.output,
                    self.input.numel(),
                    BLOCK_SIZE=2048,
                )
            else:
                tmp = torch.sin(self.input)
                tmp = tmp * self.scale
                self.output = tmp + self.bias

    def teardown(self) -> None:
        self.input = None
        self.output = None
        self.scale = None
        self.bias = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=100, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.input is None:
            return "Input tensor missing"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedTritonBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=2),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized Triton latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
