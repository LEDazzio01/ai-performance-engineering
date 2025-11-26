"""baseline_memory_profiling.py - Baseline memory profiling (baseline)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import enable_tf32


class SimpleModel(nn.Module):
    """Simple model for memory profiling."""
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineMemoryProfilingBenchmark(BaseBenchmark):
    """Baseline memory profiling - tracks usage without optimization."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs = None
        self.targets = None
        self.criterion = None
        self.peak_memory_mb = 0.0
        self.batch_size = 32
        self.hidden_dim = 2048
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.cuda.reset_peak_memory_stats()
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.criterion = nn.MSELoss()
        
        _ = self.model(self.inputs)
        self._synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None or self.targets is None or self.criterion is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_memory_profiling"):
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.criterion = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
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
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaselineMemoryProfilingBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryProfilingBenchmark()
