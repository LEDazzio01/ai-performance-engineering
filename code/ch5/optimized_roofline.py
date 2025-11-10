"""Roofline optimization: increase arithmetic intensity via fused ops."""

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
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class OptimizedRooflineBenchmark(Benchmark):
    """Uses torch.compile to fuse multiple FLOPs per byte loaded."""

    def __init__(self):
        self.device = resolve_device()
        self.data: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.kernel = None

    def setup(self) -> None:
        torch.manual_seed(2)
        enable_tf32()
        self.data = torch.randn(64_000_000, device=self.device, dtype=torch.float16)
        self.weights = torch.randn_like(self.data)

        def fused_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            acc = torch.addcmul(x, w, x)
            acc = acc * 1.5 + torch.sin(acc)
            return acc.mean()

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            try:
                self.kernel = compile_fn(fused_op, mode="reduce-overhead")
            except Exception:
                self.kernel = fused_op
        else:
            self.kernel = fused_op
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.data is not None and self.weights is not None
        with nvtx_range("optimized_roofline", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = self.kernel(self.data, self.weights)

    def teardown(self) -> None:
        self.data = None
        self.weights = None
        self.kernel = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.kernel is None:
            return "Kernel not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedRooflineBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized roofline latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
