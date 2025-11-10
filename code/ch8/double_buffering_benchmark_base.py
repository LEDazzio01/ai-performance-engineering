"""Shared base for double buffering benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.extension_loader_template import load_cuda_extension

DOUBLE_BUFFER_INNER_LOOPS = 16


class DoubleBufferingBenchmarkBase(Benchmark):
    block = 256
    tile = 4
    nvtx_label = "double_buffering"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for double buffering benchmarks")
        self.device = torch.device("cuda")
        self.extension = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.host_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch8_double_buffering_kernels",
            cuda_source_file=str(Path(__file__).with_name("double_buffering_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )

        torch.manual_seed(123)
        elements = self.block * self.tile * 8192
        device_input = torch.randn(elements, device=self.device, dtype=torch.float32).contiguous()
        self.input = device_input
        self.host_input = device_input.cpu().pin_memory()
        self.output = torch.empty_like(self.input)

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
        self.input = None
        self.output = None
        self.host_input = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _validate_correctness(self) -> None:
        assert self.input is not None
        assert self.output is not None
        reference = self.input.clone()
        for _ in range(DOUBLE_BUFFER_INNER_LOOPS):
            reference = reference * 1.0002 + reference.pow(2) * 0.00001
        torch.cuda.synchronize()
        max_error = torch.max(torch.abs(reference - self.output)).item()
        if max_error > 1e-2:
            raise RuntimeError(f"Double buffering kernel validation failed (max error={max_error:.4f})")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.output is None:
            return "Output not initialized"
        return None
