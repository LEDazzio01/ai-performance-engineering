"""baseline_multiple_unoptimized.py - Multiple unoptimized techniques (baseline).

Chapter 20: AI-Assisted Performance Optimizations

Baseline demonstrates common anti-patterns that AI optimization can fix:
1. FP32 instead of FP16/BF16 (no tensor core acceleration)
2. Redundant computation (no caching of repeated values)
3. Synchronous execution (no async/pipelining)
4. Multiple kernel launches instead of fused operations
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class UnoptimizedModel(nn.Module):
    """Model with intentional inefficiencies that AI optimization can fix."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fc3 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x):
        # Anti-pattern 1: Separate operations that could be fused
        x = self.fc1(x)
        x = torch.relu(x)  # Separate kernel
        x = self.fc2(x)
        x = torch.relu(x)  # Another separate kernel
        x = self.fc3(x)
        # Anti-pattern 2: Redundant normalization
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return x


class BaselineMultipleUnoptimizedBenchmark(BaseBenchmark):
    """Baseline: Multiple anti-patterns (FP32, unfused ops, redundant compute)."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.x: Optional[torch.Tensor] = None
        self.batch_size = 128
        self.hidden_dim = 2048  # Smaller to make differences more visible
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        # FP32 - no tensor core acceleration
        self.model = UnoptimizedModel(hidden_dim=self.hidden_dim).to(self.device).float().eval()
        self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(self.x)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.x is not None
        with self._nvtx_range("multiple_techniques_baseline"):
            with torch.no_grad():
                # Run model multiple times to simulate redundant computation
                for _ in range(3):
                    out = self.model(self.x)
                    _ = out.sum()  # Force materialization
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
        from core.benchmark.metrics import compute_ai_optimization_metrics
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
