"""baseline_training_single.py - Single-GPU training baseline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for training demonstration."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineTrainingSingleBenchmark(BaseBenchmark):
    """Single-GPU training baseline - no parallelism."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        # Heavier batch/hidden to highlight benefits of AMP/compile in the optimized path.
        self.batch_size = 32
        self.hidden_dim = 8192
        self.train_steps = 6
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).float().train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None and self.targets is not None
        assert self.optimizer is not None and self.criterion is not None
        with self._nvtx_range("training_baseline"):
            for _ in range(self.train_steps):
                self.optimizer.zero_grad()
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
                loss.backward()
                self.optimizer.step()
            self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
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
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTrainingSingleBenchmark()
