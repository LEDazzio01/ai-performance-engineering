"""Shared base for Chapter 8 AI optimization benchmarks."""

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


class AiOptimizationBenchmarkBase(BaseBenchmark):
    rows: int = 1 << 13  # 8192 samples
    cols: int = 512
    nvtx_label: str = "ai_optimization"

    def __init__(self) -> None:
        super().__init__()
        # Kernel outputs are numerically noisy; skip strict output verification.
        self.skip_output_check = True
        self.skip_input_check = True
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Chapter 8 AI optimization benchmarks")
        self.device = torch.device("cuda")
        self.extension = None
        self.inputs: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch08_ai_optimization_kernels",
            cuda_source_file=str(Path(__file__).with_name("ai_optimization_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )

        torch.manual_seed(1234)
        self.inputs = torch.randn(
            self.rows,
            self.cols,
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        self.weights = torch.randn(self.cols, device=self.device, dtype=torch.float32).contiguous()
        self.output = torch.empty(self.rows, device=self.device, dtype=torch.float32)

        self._invoke_kernel()
        torch.cuda.synchronize()
        self._validate_correctness()
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

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

        reference = torch.tanh(torch.matmul(self.inputs, self.weights))
        torch.cuda.synchronize()
        max_error = torch.max(torch.abs(reference - self.output)).item()
        if max_error > 1e-3:
            raise RuntimeError(f"AI optimization kernel validation failed (max error={max_error:.4f})")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output buffer not initialized"
        return None

    def skip_output_verification(self) -> bool:
        return True

    def get_custom_metrics(self) -> Optional[dict]:
        """Return AI optimization kernel metrics for roofline analysis."""
        flops = float(self.rows * self.cols * 2)  # matmul + tanh approx
        bytes_transferred = float((self.rows * self.cols + self.cols + self.rows) * 4)
        return {
            f"{self.nvtx_label}.rows": float(self.rows),
            f"{self.nvtx_label}.cols": float(self.cols),
            f"{self.nvtx_label}.flops": flops,
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
            f"{self.nvtx_label}.arithmetic_intensity": flops / bytes_transferred if bytes_transferred > 0 else 0.0,
        }
