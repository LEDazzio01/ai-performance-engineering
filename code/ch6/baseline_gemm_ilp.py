"""baseline_gemm_ilp.py - Baseline GEMM with low ILP (no tensor cores)."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch6.cuda_extensions import load_ilp_extension


class BaselineGEMMILPBenchmark(BaseBenchmark):
    """Baseline ILP workload using sequential CUDA kernel."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 10_000_000
        self._extension = None
        self.repeats = 4
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )
    
    def setup(self) -> None:
        """Initialize tensors and load CUDA extension."""
        self._extension = load_ilp_extension()
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()
        self._extension.sequential_ops(self.output, self.input)
        self._synchronize()
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: sequential operations (low ILP)."""
        assert self._extension is not None and self.input is not None and self.output is not None
        with self._nvtx_range("gemm_ilp_baseline"):
            src = self.input
            dst = self.output
            for _ in range(self.repeats):
                self._extension.sequential_ops(dst, src)
                src, dst = dst, src
            if src is not self.output:
                self.output.copy_(src)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        if self.output.shape[0] != self.N:
            return f"Output shape mismatch: expected {self.N}, got {self.output.shape[0]}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineGEMMILPBenchmark()
