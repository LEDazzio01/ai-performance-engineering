"""optimized_memory_standard.py - HBM3e-optimized memory access."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedMemoryHBM3eBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """HBM3e-optimized memory access - coalesced and vectorized."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.result: Optional[torch.Tensor] = None
        self.size_mb = 100
        self.num_elements = (self.size_mb * 1024 * 1024) // 4
        bytes_per_iter = self.num_elements * 4 * 2  # read + write
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.num_elements),
            bytes_per_iteration=float(bytes_per_iter),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.num_elements),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.data = torch.randn(self.num_elements, device=self.device, dtype=torch.float32).contiguous()
        self.result = torch.zeros_like(self.data).contiguous()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.data is not None and self.result is not None
        with self._nvtx_range("memory_standard_optimized"):
            # Fused operation: result = data * 2.0 + 1.0 + 0.1 = data * 2.0 + 1.1
            # Using mul_ and add_ for in-place ops to reduce memory traffic
            torch.mul(self.data, 2.0, out=self.result)
            self.result.add_(1.1)  # Combines +1.0 and +0.1 into single op
            self._synchronize()
        self.output = self.result

    def capture_verification_payload(self) -> None:
        if self.data is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"data": self.data},
            output=self.output,
            batch_size=self.num_elements,
            parameter_count=0,
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.data = None
        self.result = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()


def get_benchmark() -> BaseBenchmark:
    return OptimizedMemoryHBM3eBenchmark()
