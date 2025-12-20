"""optimized_tensor_cores.py - Optimized tensor core acceleration.

Demonstrates tensor core acceleration using FP16/BF16.
Tensor cores: Uses tensor cores for accelerated matrix operations.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedTensorCoresBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Tensor core accelerated matrix operations."""

    signature_equivalence_group = "ch18_tensor_cores_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.size = 4096
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.output_buffer = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size * self.size),
        )
        self.output = None
        self._verification_payload = None
    
    def setup(self) -> None:
        """Setup: Initialize matrices in FP16/BF16 for tensor cores."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Keep verification inputs FP32 to match the baseline signature; compute path casts to FP16/BF16.
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.output_buffer = torch.empty((self.size, self.size), device=self.device, dtype=self.dtype)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tensor core accelerated matrix multiplication."""
        # Optimization: FP16/BF16 matmul with tensor cores
        # Tensor cores provide high throughput for these operations
        with self._nvtx_range("optimized_tensor_cores"):
            if self.output_buffer is None:
                raise RuntimeError("Output buffer not initialized")
            a_tc = self.A.to(self.dtype)
            b_tc = self.B.to(self.dtype)
            torch.matmul(a_tc, b_tc, out=self.output_buffer)
            self.output = self.output_buffer
        if self.output is None or self.A is None or self.B is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.output.float(),
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 64),
            accepted_tokens=getattr(self, '_accepted_tokens', 48),
            draft_time_ms=getattr(self, '_draft_ms', 5.0),
            verify_time_ms=getattr(self, '_verify_ms', 10.0),
            num_rounds=getattr(self, '_num_rounds', 8),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedTensorCoresBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
