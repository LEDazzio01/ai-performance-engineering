"""baseline_adaptive.py - Baseline without adaptive optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineAdaptiveBenchmark(BaseBenchmark):
    """Baseline: static configuration (no runtime adaptation)."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 4_000_000
        self.static_chunk = 2048
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize with static configuration."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Shared math used by both baseline and optimized variants."""
        out = tensor.mul(1.75)
        out = out.add(0.1)
        return F.silu(out)
    
    def benchmark_fn(self) -> None:
        """Benchmark: static configuration operations."""
        assert self.input is not None and self.output is not None
        with self._nvtx_range("baseline_adaptive"):
            for start in range(0, self.N, self.static_chunk):
                end = min(start + self.static_chunk, self.N)
                window = self.input[start:end]
                transformed = self._transform(window)
                self.output[start:end].copy_(transformed)
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
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineAdaptiveBenchmark()
