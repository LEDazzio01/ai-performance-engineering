"""baseline_multiple_unoptimized.py - Multiple unoptimized techniques (baseline)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for optimization demonstration."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineMultipleUnoptimizedBenchmark(BaseBenchmark):
    """Baseline combining multiple inefficiencies (FP32, small batch, no compile, no graphs)."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.x: Optional[torch.Tensor] = None
        # Baseline: same batch size for fair comparison, but FP32 (no tensor cores)
        self.batch_size = 128
        self.hidden_dim = 4096
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).to(torch.float32).eval()
        self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.x is not None
        with self._nvtx_range("multiple_techniques_baseline"):
            with torch.no_grad():
                _ = self.model(self.x)
            self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.x = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
        )
    
    def get_workload_metadata(self):
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMultipleUnoptimizedBenchmark()
