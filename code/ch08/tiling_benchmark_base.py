"""Shared utilities for ch08 tiling benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension

def resolve_device() -> torch.device:
    """Return the CUDA device used for Chapter 8 examples."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Chapter 8 tiling benchmarks")
    return torch.device("cuda")


class TilingBenchmarkBase(BaseBenchmark):
    """Base class that pre-loads the CUDA tiling extension and inputs."""

    extension_name = "ch08_tiling_kernels"
    kernel_source = Path(__file__).with_name("tiling_kernels.cu")
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]
    extra_ldflags = ["-lcublas"]
    tensor_dtype = torch.float32

    nvtx_label: str = "tiling"
    matrix_rows: int = 2048
    matrix_cols: int = 2048
    shared_dim: int = 2048

    def __init__(self) -> None:
        super().__init__()
        self.skip_output_check = True
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
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

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
        # Large GEMMs accumulate floating-point error quickly; tolerate small
        # absolute differences that stem from reordering in the tiled kernel.
        if max_error > 1e-1:
            raise RuntimeError(
                f"Tiling kernel validation failed (max error={max_error:.4f})"
            )

    # --------------------------------------------------------------------- #
    # Benchmark configuration
    # --------------------------------------------------------------------- #
    def get_config(self) -> BenchmarkConfig:
        """Use fewer iterations because each kernel is compute-heavy."""
        return BenchmarkConfig(
            iterations=48,
            warmup=8,
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

    def skip_output_verification(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Extension loading (allow subclasses to override)
    # ------------------------------------------------------------------ #
    def _load_extension(self) -> None:
        """Compile/load the default CUDA extension."""
        self.extension = load_cuda_extension(
            extension_name=self.extension_name,
            cuda_source_file=str(self.kernel_source),
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return tiling optimization metrics for roofline analysis."""
        M, K, N = self.matrix_rows, self.shared_dim, self.matrix_cols
        flops = 2.0 * M * K * N  # MAD operations
        bytes_transferred = (M * K + K * N + M * N) * 4.0  # float32
        return {
            f"{self.nvtx_label}.matrix_m": float(M),
            f"{self.nvtx_label}.matrix_k": float(K),
            f"{self.nvtx_label}.matrix_n": float(N),
            f"{self.nvtx_label}.flops": flops,
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
            f"{self.nvtx_label}.arithmetic_intensity": flops / bytes_transferred if bytes_transferred > 0 else 0.0,
        }
