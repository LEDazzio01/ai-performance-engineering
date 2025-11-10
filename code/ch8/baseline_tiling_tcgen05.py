"""Baseline tiling benchmark gated for tcgen05 comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch8.baseline_tiling import BaselineTilingBenchmark
from common.python.tcgen05_requirements import ensure_tcgen05_supported
from common.tcgen05 import load_tiling_tcgen05_module


class BaselineTilingBenchmarkTCGen05(BaselineTilingBenchmark):
    """tcgen05 baseline that drives the CUTLASS inline kernel directly."""

    nvtx_label = "baseline_tiling_tcgen05"

    def __init__(self) -> None:
        ensure_tcgen05_supported(
            loader=load_tiling_tcgen05_module,
            module_name="ch8 tiling tcgen05 kernels",
        )
        self.tcgen05_extension = load_tiling_tcgen05_module()
        # tcgen05 kernels require fp16 operands.
        self.tensor_dtype = torch.float16
        super().__init__()

    def _load_extension(self) -> None:
        """Skip the fp32 baseline extension and reuse the tcgen05 module."""
        self.extension = self.tcgen05_extension

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None
        result = self.extension.matmul_tiling_tcgen05(self.matrix_a, self.matrix_b)
        self.output.copy_(result)

    def _validate_correctness(self) -> None:
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None
        with torch.no_grad():
            reference = torch.matmul(
                self.matrix_a.to(torch.float32), self.matrix_b.to(torch.float32)
            ).to(self.output.dtype)
        torch.cuda.synchronize()
        max_error = torch.max(torch.abs(self.output - reference)).item()
        if max_error > 2e-2:
            raise RuntimeError(
                f"tcgen05 tiling kernel validation failed (max error={max_error:.4f})"
            )


def get_benchmark() -> BaselineTilingBenchmarkTCGen05:
    return BaselineTilingBenchmarkTCGen05()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nBaseline tcgen05 tiling GEMM: "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
