"""Shared utilities for ch8 tiling benchmarks."""

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

def resolve_device() -> torch.device:
    """Return the CUDA device used for Chapter 8 examples."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Chapter 8 tiling benchmarks")
    return torch.device("cuda")


class TilingBenchmarkBase(Benchmark):
    """Base class that pre-loads the CUDA tiling extension and inputs."""

    extension_name = "ch8_tiling_kernels"
    kernel_source = Path(__file__).with_name("tiling_kernels.cu")
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]
    tensor_dtype = torch.float32

    nvtx_label: str = "tiling"
    matrix_rows: int = 2048
    matrix_cols: int = 2048
    shared_dim: int = 2048

    def __init__(self) -> None:
        self.device = resolve_device()
        self.extension = None
        self.matrix_a: Optional[torch.Tensor] = None
        self.matrix_b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    # --------------------------------------------------------------------- #
    # Benchmark lifecycle
    # --------------------------------------------------------------------- #
    def setup(self) -> None:
        """Compile/load the CUDA extension, allocate tensors, and warm up."""
        self._load_extension()

        torch.manual_seed(42)
        self.matrix_a = torch.randn(
            self.matrix_rows,
            self.shared_dim,
            device=self.device,
            dtype=self.tensor_dtype,
        )
        self.matrix_b = torch.randn(
            self.shared_dim,
            self.matrix_cols,
            device=self.device,
            dtype=self.tensor_dtype,
        )
        self.output = torch.empty(
            self.matrix_rows,
            self.matrix_cols,
            device=self.device,
            dtype=self.tensor_dtype,
        )

        # Warm up + trigger first-time compilation outside measurement.
        self._invoke_kernel()
        torch.cuda.synchronize()

        # Validate correctness once so benchmark iterations can skip it.
        self._validate_correctness()
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Run the core kernel with NVTX labeling."""
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.nvtx_label, enable=enable_nvtx):
            self._invoke_kernel()

    def teardown(self) -> None:
        """Release GPU memory."""
        self.matrix_a = None
        self.matrix_b = None
        self.output = None
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _invoke_kernel(self) -> None:
        """Call into the CUDA extension (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _invoke_kernel()")

    def _validate_correctness(self) -> None:
        """Ensure the CUDA kernel matches torch.matmul for the same inputs."""
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None

        with torch.no_grad():
            reference = torch.matmul(self.matrix_a, self.matrix_b)
        torch.cuda.synchronize()

        max_error = torch.max(torch.abs(self.output - reference)).item()
        if max_error > 1e-2:
            raise RuntimeError(
                f"Tiling kernel validation failed (max error={max_error:.4f})"
            )

    # --------------------------------------------------------------------- #
    # Benchmark configuration
    # --------------------------------------------------------------------- #
    def get_config(self) -> BenchmarkConfig:
        """Use fewer iterations because each kernel is compute-heavy."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Verify tensors are present before the harness records results."""
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.matrix_a is None or self.matrix_b is None:
            return "Input matrices not initialized"
        if self.output is None:
            return "Output buffer not initialized"
        return None

    # ------------------------------------------------------------------ #
    # Extension loading (allow subclasses to override)
    # ------------------------------------------------------------------ #
    def _load_extension(self) -> None:
        """Compile/load the default CUDA extension."""
        self.extension = load_cuda_extension(
            extension_name=self.extension_name,
            cuda_source_file=str(self.kernel_source),
            extra_cuda_cflags=self.extra_cuda_cflags,
        )
