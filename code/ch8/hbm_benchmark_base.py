"""Shared utilities for Chapter 8 HBM benchmarks."""

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


class HBMBenchmarkBase(Benchmark):
    rows: int = 4096
    cols: int = 2048
    nvtx_label: str = "hbm"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for HBM benchmarks")
        self.device = torch.device("cuda")
        self.extension = None
        self.matrix_row: Optional[torch.Tensor] = None
        self.matrix_col: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.host_col: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch8_hbm_kernels",
            cuda_source_file=str(Path(__file__).with_name("hbm_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )
        torch.manual_seed(42)
        host_row = torch.randn(
            self.rows,
            self.cols,
            dtype=torch.float32,
        )
        host_col = host_row.transpose(0, 1).contiguous()

        self.host_col = host_col.pin_memory()
        self.matrix_row = host_row.to(self.device, non_blocking=False).contiguous()
        self.matrix_col = self.host_col.to(self.device, non_blocking=False).contiguous()
        self.output = torch.empty(self.rows, device=self.device, dtype=torch.float32)

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
        self.matrix_row = None
        self.matrix_col = None
        self.output = None
        self.host_col = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _mix_reference(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 1.00005 + torch.sin(tensor) * 0.00095 + torch.cos(tensor) * 0.00073

    def _validate_correctness(self) -> None:
        assert self.matrix_row is not None
        assert self.output is not None

        reference = self._mix_reference(self.matrix_row).sum(dim=1)
        torch.cuda.synchronize()
        max_error = torch.max(torch.abs(reference - self.output)).item()
        if max_error > 5e-3:
            raise RuntimeError(f"HBM kernel validation failed (max error={max_error:.4f})")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.output is None:
            return "Output buffer not initialized"
        return None
