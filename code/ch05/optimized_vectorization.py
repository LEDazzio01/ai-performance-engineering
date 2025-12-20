"""optimized_vectorization.py - Optimized vectorized operations in storage I/O context."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedVectorizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Vectorized operations for efficient processing."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 1_000_000
        # Computation benchmark - jitter check not applicable
        tokens = self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        # Enable cuDNN benchmarking for optimal kernel selection
        torch.manual_seed(42)
        self.data = torch.randn(self.N, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Vectorized sum of first 1000 elements (same as baseline)."""
        assert self.data is not None
        with self._nvtx_range("optimized_vectorization"):
            # Vectorized sum of first 1000 elements to match baseline
            result = self.data[:1000].sum().unsqueeze(0)
            self._synchronize()
        self.output = result

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"data": self.data},
            output=self.output.detach().clone(),
            batch_size=self.data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedVectorizationBenchmark()
