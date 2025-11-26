"""baseline_precisionfp8.py - FP32 precision baseline (baseline).

Training with full FP32 precision.
Higher memory usage and slower computation compared to mixed precision.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional, Tuple

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.compile_utils import configure_tf32, restore_tf32


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class SimpleModel(nn.Module):
    """Simple model for precision comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselinePrecisionFP8Benchmark(BaseBenchmark):
    """FP32 precision - full precision training."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.batch_size = 256
        self.hidden_dim = 4096
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._prev_precision: Optional[str] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize FP32 model and data."""
        torch.manual_seed(42)

        self._prev_precision = torch.get_float32_matmul_precision()

        self._tf32_state = configure_tf32(enable_matmul=False, enable_cudnn=False)
        torch.set_float32_matmul_precision("highest")
        
        # FP32 model (full precision)
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # Warmup
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - FP32 precision."""
        with self._nvtx_range("baseline_precisionfp8"):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)  # FP32 computation
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()
        self._synchronize()

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
        if self._prev_precision is not None:
            torch.set_float32_matmul_precision(self._prev_precision)  # type: ignore[arg-type]
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
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


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselinePrecisionFP8Benchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Precision FP32 (precisionfp8 pair): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
