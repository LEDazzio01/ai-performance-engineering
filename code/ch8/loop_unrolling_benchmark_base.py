"""Shared utilities for loop unrolling benchmarks."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.extension_loader_template import load_cuda_extension

_KERNEL_SOURCE = Path(__file__).with_name("loop_unrolling_kernels.cu")
_EXTENSION_NAME = "ch8_loop_unrolling_kernels"


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Chapter 8 loop-unrolling benchmarks")
    return torch.device("cuda")


class LoopUnrollingBenchmarkBase(Benchmark):
    """Base class that manages CUDA extension loading and tensor setup."""

    rows: int = 1 << 14  # 16,384 rows
    elements_per_row: int = 512
    weight_period: int = 8
    nvtx_label: str = "loop_unrolling"

    def __init__(self) -> None:
        self.device = resolve_device()
        self.extension = None
        self.inputs: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name=_EXTENSION_NAME,
            cuda_source_file=str(_KERNEL_SOURCE),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )

        torch.manual_seed(42)
        self.inputs = torch.randn(
            self.rows,
            self.elements_per_row,
            device=self.device,
            dtype=torch.float32,
        )
        self.weights = torch.randn(
            self.weight_period,
            device=self.device,
            dtype=torch.float32,
        )
        self.output = torch.empty(
            self.rows,
            device=self.device,
            dtype=torch.float32,
        )

        # Warm up + trigger compilation outside measurement window.
        self._invoke_kernel()
        torch.cuda.synchronize()
        self._validate_correctness()
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.nvtx_label, enable=enable_nvtx):
            self._invoke_kernel()

    def teardown(self) -> None:
        self.inputs = None
        self.weights = None
        self.output = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _validate_correctness(self) -> None:
        assert self.inputs is not None
        assert self.weights is not None
        assert self.output is not None

        repeats = math.ceil(self.elements_per_row / self.weight_period)
        tiled_weights = self.weights.repeat(repeats)[: self.elements_per_row]
        reference = (self.inputs * tiled_weights).sum(dim=1)
        torch.cuda.synchronize()
        max_error = torch.max(torch.abs(reference - self.output)).item()
        if max_error > 5e-3:
            raise RuntimeError(
                f"Loop unrolling kernel validation failed (max error={max_error:.4f})"
            )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.inputs is None or self.weights is None:
            return "Inputs not initialized"
        if self.output is None:
            return "Output buffer not initialized"
        return None
