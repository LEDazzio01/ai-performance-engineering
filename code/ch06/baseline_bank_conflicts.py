"""baseline_bank_conflicts.py - Shared memory bank conflicts (baseline)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.cuda_extensions import load_bank_conflicts_extension


class BaselineBankConflictsBenchmark(BaseBenchmark):
    """Bank conflicts - poor shared memory access pattern (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 8_000_000
        self._extension = None
        self.repeats = 8
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )
    
    def setup(self) -> None:
        """Initialize tensors and load CUDA extension."""
        self._extension = load_bank_conflicts_extension()
        
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()
        # Warm up to exclude compile/setup cost.
        self._extension.bank_conflicts(self.output, self.input)
        self._synchronize()
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: access pattern causing bank conflicts."""
        assert self._extension is not None and self.output is not None and self.input is not None
        with self._nvtx_range("bank_conflicts_baseline"):
            for _ in range(self.repeats):
                self._extension.bank_conflicts(self.output, self.input)
            self._synchronize()
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        if self.input is None:
            return "Input tensor not initialized"
        if self.output.shape != self.input.shape:
            return f"Shape mismatch: input={self.input.shape}, output={self.output.shape}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineBankConflictsBenchmark()
