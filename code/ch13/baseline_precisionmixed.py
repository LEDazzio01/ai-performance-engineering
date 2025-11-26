"""baseline_precisionmixed.py - FP32 baseline for mixed-precision comparison."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import configure_tf32, restore_tf32


class SimpleModel(nn.Module):
    """Deeper MLP to stress GEMM throughput."""

    def __init__(self, hidden_dim: int = 1024, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 2) for _ in range(depth)
        ])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_proj(x))
        for layer in self.hidden:
            x = self.act(layer(x))
        x = self.out_proj(x)
        return x


class BaselinePrecisionMixedBenchmark(BaseBenchmark):
    """FP32 baseline used to compare against mixed-precision autocast."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.batch_size = 128
        self.hidden_dim = 2048
        self.micro_steps = 4
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_steps),
            tokens_per_iteration=float(tokens * self.micro_steps),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            self._tf32_state = configure_tf32(enable_matmul=False, enable_cudnn=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn_like(self.inputs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - FP32 training."""
        if any(v is None for v in (self.model, self.inputs, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_precision_mixed"):
            for _ in range(self.micro_steps):
                self.optimizer.zero_grad(set_to_none=True)
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
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
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


def get_benchmark() -> BaselinePrecisionMixedBenchmark:
    """Factory function for harness discovery."""
    return BaselinePrecisionMixedBenchmark()
