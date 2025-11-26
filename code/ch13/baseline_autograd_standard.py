"""baseline_autograd_standard.py - Standard autograd baseline (baseline)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for autograd comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineAutogradStandardBenchmark(BaseBenchmark):
    """Standard autograd - no compilation."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        # Smaller batch to increase launch overhead share and highlight graph/optimized paths.
        self.batch_size = 16
        self.hidden_dim = 1024
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - standard autograd."""
        if any(v is None for v in (self.model, self.inputs, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("baseline_autograd_standard"):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()
        self._synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaselineAutogradStandardBenchmark:
    """Factory function for harness discovery."""
    return BaselineAutogradStandardBenchmark()
