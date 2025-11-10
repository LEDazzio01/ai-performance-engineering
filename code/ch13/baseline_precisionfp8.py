"""baseline_precisionfp8.py - FP32 precision baseline (baseline).

Training with full FP32 precision.
Higher memory usage and slower computation compared to mixed precision.

Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


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


class BaselinePrecisionFP8Benchmark(Benchmark):
    """FP32 precision - full precision training."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.batch_size = 256
        self.hidden_dim = 4096
        self._prev_tf32_matmul: Optional[bool] = None
        self._prev_tf32_cudnn: Optional[bool] = None
        self._prev_precision: Optional[str] = None
    
    def setup(self) -> None:
        """Setup: Initialize FP32 model and data."""
        torch.manual_seed(42)

        self._prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        self._prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
        self._prev_precision = torch.get_float32_matmul_precision()

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
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
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - FP32 precision."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_precisionfp8", enable=enable_nvtx):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)  # FP32 computation
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion
        if self._prev_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = self._prev_tf32_matmul
        if self._prev_tf32_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = self._prev_tf32_cudnn
        if self._prev_precision is not None:
            torch.set_float32_matmul_precision(self._prev_precision)  # type: ignore[arg-type]
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
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
