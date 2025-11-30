"""Benchmark wrapper for the optimized CUDA decode kernel."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.harness.cuda_capabilities import tma_support_status
from core.harness.hardware_capabilities import detect_capabilities
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

from labs.moe_cuda.decode_kernels import (
    run_baseline_kernel,
    optimized_kernel_supported,
    run_optimized_kernel,
    is_optimized_available,
    get_optimized_error,
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
        import gc
        
        # CRITICAL: Comprehensive CUDA cleanup before TMA kernel setup
        # TMA tensor map encoding is very sensitive to CUDA memory/graph state
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Reset CUDA graph pool
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # Reset CUDA RNG state - this is CRITICAL
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(7)
        except Exception:
            pass
        
        # Reset dynamo/inductor state
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        
        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass
        
        torch.manual_seed(7)
        
        # Allocate contiguous tensors with explicit memory layout
        # TMA requires contiguous tensors with proper alignment
        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.input = torch.randn(
            self.rows,
            self.cols,
            dtype=torch.float32,
        ).to(self.device).contiguous()  # Explicitly ensure contiguity
        
        self.output = torch.zeros(
            self.rows,
            self.cols,
            dtype=torch.float32,
        ).to(self.device).contiguous()  # Explicitly ensure contiguity
        
        torch.cuda.synchronize(self.device)
        
        # Verify tensors are properly allocated before benchmark
        assert self.input.is_contiguous(), "Input tensor must be contiguous for TMA"
        assert self.output.is_contiguous(), "Output tensor must be contiguous for TMA"

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
    """Return the optimized TMA decode kernel benchmark.
    
    TMA is required on Blackwell B200 - no fallbacks.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for TMA decode kernel")
    
    supported, reason = tma_support_status()
    if not supported:
        raise RuntimeError(f"TMA decode kernel unavailable: {reason}")
    
    cap = detect_capabilities()
    if cap is None:
        raise RuntimeError("TMA decode kernel requires detected hardware capabilities")
    
    cap_desc = f"{cap.device_name} ({cap.compute_capability})"
    
    # Check if optimized kernel is available
    if not is_optimized_available():
        error = get_optimized_error() or "Unknown error"
        raise RuntimeError(f"TMA optimized kernel not available: {error}")
    
    candidate = OptimizedDecodeKernelBenchmark()
    
    # Verify TMA support for this shape
    if not optimized_kernel_supported(candidate.rows, candidate.cols):
        raise RuntimeError(
            f"TMA decode kernel not supported for shape ({candidate.rows}, {candidate.cols}) "
            f"on {cap_desc}. Check CUDA driver/runtime."
        )
    
    return candidate
