"""Optimized Triton benchmark for Chapter 7 memory experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python import triton_compat  # noqa: F401

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")


if TRITON_AVAILABLE:

    @triton.jit
    def residual_relu_kernel(
        dst_ptr,
        src_ptr,
        residual_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        combined = tl.maximum(values + residual, 0.0)
        tl.store(dst_ptr + offsets, combined, mask=mask)

    def fused_residual_relu(dst: Tensor, src: Tensor, residual: Tensor) -> Tensor:
        """Launch Triton fused residual add + ReLU kernel."""
        assert dst.is_cuda and src.is_cuda and residual.is_cuda
        assert dst.numel() == src.numel() == residual.numel()
        n_elements = dst.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
        residual_relu_kernel[grid](  # type: ignore[misc]
            dst,
            src,
            residual,
            n_elements,
            BLOCK_SIZE=4096,
            num_warps=4,
            num_stages=2,
        )
        return dst


class OptimizedTritonBenchmark(Benchmark):
    """Optimized path that fuses residual add + activation using Triton."""

    def __init__(self) -> None:
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.large_batch: Optional[Tensor] = None
        self.residual: Optional[Tensor] = None
        self.output: Optional[Tensor] = None
        self.use_triton = TRITON_AVAILABLE
        self._checksum = 0.0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.model = nn.Sequential(
            nn.Linear(1024, 2048, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(2048, 1024, bias=False),
        ).to(self.device).eval()

        total_samples = 64 * 8  # Match baseline's cumulative micro-batches.
        self.large_batch = torch.randn(total_samples, 1024, device=self.device)
        self.residual = torch.randn_like(self.large_batch)
        self.output = torch.empty_like(self.large_batch)

        if self.use_triton:
            fused_residual_relu(self.output, self.large_batch, self.residual)

        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.model is not None
        assert self.large_batch is not None
        assert self.residual is not None
        assert self.output is not None

        with nvtx_range("triton_optimized", enable=enable_nvtx):
            with torch.no_grad():
                activations = self.model(self.large_batch)
                if self.use_triton and TRITON_AVAILABLE:
                    fused_residual_relu(self.output, activations, self.residual)
                else:
                    torch.add(
                        activations,
                        self.residual,
                        alpha=1.0,
                        out=self.output,
                    )
                    torch.relu_(self.output)
                self._checksum = float(self.output.sum().item())

    def teardown(self) -> None:
        self.model = None
        self.large_batch = None
        self.residual = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.large_batch is None or self.residual is None or self.output is None:
            return "Buffers not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedTritonBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5),
    )
    benchmark = OptimizedTritonBenchmark()
    result = harness.benchmark(benchmark)

    print("=" * 70)
    print("Optimized: Triton fused residual")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
