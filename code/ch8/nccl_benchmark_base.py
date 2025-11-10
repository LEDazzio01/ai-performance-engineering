"""Shared base for NCCL-style reduction benchmarks."""

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


class NcclBenchmarkBase(Benchmark):
    world_size: int = 4
    chunk_elems: int = 1 << 15
    nvtx_label: str = "nccl"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for NCCL benchmarks")
        self.device = torch.device("cuda")
        self.extension = None
        self.device_chunks: Optional[torch.Tensor] = None
        self.host_chunks: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch8_nccl_kernels",
            cuda_source_file=str(Path(__file__).with_name("nccl_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )
        torch.manual_seed(7)
        self.device_chunks = torch.randn(
            self.world_size,
            self.chunk_elems,
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        self.host_chunks = self.device_chunks.cpu().pin_memory()
        self.output = torch.empty(self.chunk_elems, device=self.device, dtype=torch.float32)

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
        self.device_chunks = None
        self.host_chunks = None
        self.output = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _validate_correctness(self) -> None:
        assert self.device_chunks is not None
        assert self.output is not None
        reference = self.device_chunks.sum(dim=0)
        torch.cuda.synchronize()
        max_error = torch.max(torch.abs(reference - self.output)).item()
        if max_error > 1e-3:
            raise RuntimeError(f"NCCL reduction validation failed (max error={max_error:.4f})")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not initialized"
        return None

