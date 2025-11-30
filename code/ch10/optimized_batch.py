"""optimized_batch.py - Optimized large batch size in GEMM context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.utils.compile_utils import enable_tf32
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch10.workload_config import WORKLOAD


class OptimizedBatchBenchmark(BaseBenchmark):
    """Optimized: large batch size to maximize GPU utilization.
    
    Processes all data in a single forward pass, achieving better
    GPU utilization through larger matrix operations.
    """
    
    def __init__(self):
        super().__init__()
        self.model: nn.Sequential | None = None
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.workload = WORKLOAD
        self.total_batch_size = self.workload.optimized_batch_size  # 512
        self.hidden_dim = self.workload.hidden_dim
        self.ffn_dim = self.workload.ffn_dim
        tokens = self.total_batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model with optimized batch size."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        # Harness provides seeding - model and input creation order must match baseline
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        # Generate input (same shape/order as baseline for verification)
        self.input = torch.randn(self.total_batch_size, self.hidden_dim, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with optimized batch size (single kernel launch)."""
        if self.model is None or self.input is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("batch_optimized"):
            with torch.no_grad():
                # Single large forward pass (one kernel launch, better GPU utilization)
                self.output = self.model(self.input)
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.output = None
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
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

    def get_output_for_verification(self) -> Optional[torch.Tensor]:
        """Return output tensor for verification against baseline version."""
        return self.output
    
    def get_output_tolerance(self) -> tuple[float, float]:
        """Return (rtol, atol) for output verification.
        
        Large-batch vs micro-batching can have numerical differences due to:
        - Different CUDA kernel parallelism patterns
        - Different floating-point reduction order
        - TF32 precision accumulation across multiple operations
        """
        return (0.05, 0.05)  # 5% relative tolerance for batch size comparisons


def get_benchmark() -> OptimizedBatchBenchmark:
    """Factory function for harness discovery."""
    return OptimizedBatchBenchmark()
