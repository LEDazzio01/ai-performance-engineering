"""optimized_roofline_quantization.py - GPU roofline-aware quantization."""

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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch14")
    return torch.device("cuda")


class OptimizedRooflineQuantizationBenchmark(Benchmark):
    """Optimized: quantize/dequantize entirely on GPU and record roofline stats."""

    def __init__(self):
        self.device = resolve_device()
        self.tensor: torch.Tensor | None = None
        self.num_chunks = 32
        self.chunk_len = 1 << 13
        self._last = 0.0
        self.roofline_data: dict[str, float | bool] = {}

    def setup(self) -> None:
        torch.manual_seed(42)
        self.tensor = torch.randn(self.num_chunks, self.chunk_len, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_roofline_quantization", enable=enable_nvtx):
            if self.tensor is None:
                raise RuntimeError("Tensor not initialized")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            max_abs = torch.amax(self.tensor.abs(), dim=1, keepdim=True).clamp(min=1e-6)
            scales = 127.0 / max_abs
            quantized = torch.clamp(torch.round(self.tensor * scales), -127, 127).to(torch.int8)
            dequant = quantized.float() / scales
            self._last = float(dequant.sum())
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            bytes_moved = (
                self.tensor.element_size() * self.tensor.numel()
                + quantized.element_size() * quantized.numel()
            )
            compute_ops = float(self.tensor.numel())
            self.roofline_data = {
                "elapsed_ms": elapsed_ms,
                "arithmetic_intensity": compute_ops / bytes_moved if bytes_moved else 0.0,
                "memory_bound": True,
                "optimization_applied": True,
            }

    def teardown(self) -> None:
        self.tensor = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        if not self.roofline_data:
            return "Roofline data missing"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedRooflineQuantizationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Roofline Quantization: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Quantize/dequantize stays on GPU to stay compute bound")
