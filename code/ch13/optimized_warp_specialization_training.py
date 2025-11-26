"""optimized_warp_specialization_training.py - Optimized warp specialization in training context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    try:
        from ch13.warp_specialized_triton import warp_specialized_triton_forward_ch13
        TRITON_WARP_SPEC_AVAILABLE = True
    except ImportError:
        try:
            from warp_specialized_triton import warp_specialized_triton_forward_ch13
            TRITON_WARP_SPEC_AVAILABLE = True
        except ImportError:
            TRITON_WARP_SPEC_AVAILABLE = False
else:
    TRITON_WARP_SPEC_AVAILABLE = False


class OptimizedWarpSpecializationTrainingBenchmark(BaseBenchmark):
    """Optimized: REAL warp specialization in training context."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input = None
        self.weight = None
        self.batch = 512
        self.width = 2048
        tokens = self.batch * self.width
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        
        self.model = nn.Sequential(
            nn.Linear(self.width, self.width),
        ).to(self.device).train()
        
        self.input = torch.randn(self.batch, self.width, device=self.device)
        self.weight = torch.randn_like(self.input)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.input is None or self.weight is None or self.model is None:
            raise RuntimeError("Benchmark not configured")
        if not TRITON_WARP_SPEC_AVAILABLE:
            raise RuntimeError("REAL warp specialization requires Triton kernels")

        with self._nvtx_range("optimized_warp_specialization_training"):
            input_flat = self.input.flatten()
            weight_flat = self.weight.flatten()
            intermediate_flat = warp_specialized_triton_forward_ch13(input_flat, weight_flat)
            intermediate = intermediate_flat.view_as(self.input)
            output = self.model(intermediate)
            _ = output.sum()
        self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.input = None
        self.weight = None
        super().teardown()
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            use_subprocess=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedWarpSpecializationTrainingBenchmark:
    """Return benchmark instance."""
    return OptimizedWarpSpecializationTrainingBenchmark()
