"""Shared threshold benchmark utilities for Chapter 8."""

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

THRESHOLD_SECONDARY_SCALE = 1.5
THRESHOLD_INNER_SCALE = 0.85
THRESHOLD_OUTER_SCALE = 1.25


class ThresholdBenchmarkBase(BaseBenchmark):
    rows: int = 1 << 24  # 16M elements - larger to show memory transfer impact
    threshold: float = 2.5
    nvtx_label: str = "threshold"

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Chapter 8 threshold benchmarks")
        self.device = torch.device("cuda")
        self.inputs: Optional[torch.Tensor] = None
        self.outputs: Optional[torch.Tensor] = None
        self.host_inputs: Optional[torch.Tensor] = None
        self.extension = None

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch08_threshold_kernels",
            cuda_source_file=str(Path(__file__).with_name("threshold_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )

        torch.manual_seed(42)
        device_inputs = self._generate_inputs()
        self.inputs = device_inputs.contiguous()
        self.host_inputs = self.inputs.cpu().pin_memory()
        self.outputs = torch.empty_like(self.inputs)

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
        self.outputs = None
        self.host_inputs = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _generate_inputs(self) -> torch.Tensor:
        return torch.randn(self.rows, device=self.device, dtype=torch.float32) * 0.25

    def _validate_correctness(self) -> None:
        assert self.inputs is not None
        assert self.outputs is not None

        sine = torch.sin(self.inputs)
        cosine = torch.cos(self.inputs)
        magnitude = self.inputs.abs() + sine * cosine * 0.0001
        abs_inputs = self.inputs.abs()
        active = abs_inputs > self.threshold
        outer = abs_inputs > (self.threshold * THRESHOLD_SECONDARY_SCALE)

        inner_scale = torch.full_like(self.inputs, THRESHOLD_INNER_SCALE)
        outer_scale = torch.full_like(self.inputs, THRESHOLD_OUTER_SCALE)
        scale = torch.where(outer, outer_scale, inner_scale)
        signed_scale = torch.where(self.inputs >= 0, scale, -scale)
        reference = torch.where(
            active,
            magnitude * signed_scale,
            torch.zeros_like(self.inputs),
        )
        torch.cuda.synchronize()
        max_error = torch.max(torch.abs(reference - self.outputs)).item()
        if max_error > 5e-3:
            raise RuntimeError(
                f"Threshold kernel validation failed (max error={max_error:.4f})"
            )

    def get_config(self) -> BenchmarkConfig:
        # Reduce runtime so full benchmark runs avoid harness watchdog on large data sets.
        return BenchmarkConfig(iterations=10, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.inputs is None or self.outputs is None:
            return "Buffers not initialized"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return threshold kernel optimization metrics."""
        bytes_transferred = float(self.rows * 4 * 2)  # float32 read + write
        return {
            f"{self.nvtx_label}.rows": float(self.rows),
            f"{self.nvtx_label}.threshold": self.threshold,
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
        }
