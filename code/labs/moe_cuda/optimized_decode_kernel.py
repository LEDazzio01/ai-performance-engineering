"""Benchmark wrapper for the optimized CUDA decode kernel."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

from labs.moe_cuda.decode_kernels import (
    run_baseline_kernel,
    optimized_kernel_supported,
    run_optimized_kernel,
)


class OptimizedDecodeKernelBenchmark(BaseBenchmark):
    """Runs the TMA double-buffered CUDA decode kernel."""

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda decode kernels require CUDA")
        self.rows = 4096
        self.cols = 1024
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.rows * self.cols
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(7)
        self.input = torch.randn(
            self.rows,
            self.cols,
            device=self.device,
            dtype=torch.float32,
        )
        self.output = torch.zeros_like(self.input)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("Decode tensors not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_decode_kernel_optimized", enable=enable_nvtx):
            run_optimized_kernel(self.input, self.output)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.input = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5, measurement_timeout_seconds=60, setup_timeout_seconds=60)  # Min warmup for CUDA

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "decode_kernel.estimated_flops": flops,
            "decode_kernel.estimated_bytes": bytes_moved,
            "decode_kernel.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.output is None:
            return "Decode tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    candidate = OptimizedDecodeKernelBenchmark()
    arch = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    supports_optimized = False
    # Avoid expensive extension compilation on SM12x where tcgen05/TMA paths are
    # currently unreliable; fall back to the baseline kernel instead.
    if arch[0] < 12:
        supports_optimized = optimized_kernel_supported(candidate.rows, candidate.cols)
    if supports_optimized:
        return candidate

    class FallbackBenchmark(OptimizedDecodeKernelBenchmark):
        """Fallback to baseline kernel when TMA path is unavailable on this stack."""

        def benchmark_fn(self) -> None:  # pragma: no cover - benchmarked
            if self.input is None or self.output is None:
                raise RuntimeError("Decode tensors not initialized")
            enable_nvtx = get_nvtx_enabled(self.get_config())
            with nvtx_range("moe_cuda_decode_kernel_baseline_fallback", enable=enable_nvtx):
                run_baseline_kernel(self.input, self.output)
            torch.cuda.synchronize(self.device)

    return FallbackBenchmark()
