"""Benchmark wrapper for the baseline CUDA decode kernel."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

from labs.moe_cuda.decode_kernels import run_baseline_kernel


class BaselineDecodeKernelBenchmark(BaseBenchmark):
    """Runs the naive global-load CUDA decode kernel."""

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
        import gc
        
        # Clean up CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # Reset CUDA RNG state to prevent graph capture errors
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(7)
        except Exception:
            pass
        
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        
        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass
        
        torch.manual_seed(7)
        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.input = torch.randn(
            self.rows,
            self.cols,
            dtype=torch.float32,
        ).to(self.device)
        self.output = torch.zeros_like(self.input)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("Decode tensors not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_decode_kernel_baseline", enable=enable_nvtx):
            run_baseline_kernel(self.input, self.output)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.input = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        # Use shorter runs to keep verification fast on slow builds/GPUs.
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

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "shapes": {
                "input": (self.rows, self.cols),
                "output": (self.rows, self.cols),
            },
            "dtypes": {"input": "float32"},
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeKernelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
