"""baseline_batch.py - Baseline small batch size in GEMM context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch10.workload_config import WORKLOAD


class BaselineBatchBenchmark(BaseBenchmark):
    """Baseline: small batch size, limited GPU utilization."""
    
    def __init__(self):
        super().__init__()
        self.model: nn.Sequential | None = None
        self.inputs: torch.Tensor | None = None
        self.workload = WORKLOAD
        self.micro_batch_size = self.workload.baseline_micro_batch_size
        self.micro_batches = self.workload.baseline_micro_batches
        self.hidden_dim = self.workload.hidden_dim
        self.ffn_dim = self.workload.ffn_dim
        tokens = self.micro_batches * self.micro_batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )
        self._last_total = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model with small batch size."""
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        self.inputs = torch.randn(
            self.micro_batches,
            self.micro_batch_size,
            self.hidden_dim,
            device=self.device,
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with small batch size."""
        if self.model is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("batch_baseline"):
            total = 0.0
            for idx in range(self.micro_batches):
                output = self.model(self.inputs[idx])
                total += float(output.sum())
            self._last_total = total
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
        super().teardown()
    
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
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> BaselineBatchBenchmark:
    """Factory function for harness discovery."""
    return BaselineBatchBenchmark()
