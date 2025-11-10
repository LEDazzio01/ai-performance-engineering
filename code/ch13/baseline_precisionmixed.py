"""baseline_precisionmixed.py - FP32 baseline for mixed-precision comparison."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.optim import Optimizer


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


class BaselinePrecisionMixedBenchmark(Benchmark):
    """FP32 baseline used to compare against mixed-precision autocast."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.batch_size = 128
        self.hidden_dim = 2048
        self.micro_steps = 4
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn_like(self.inputs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # Warmup
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - BF16 training."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_precision_mixed", enable=enable_nvtx):
            for _ in range(self.micro_steps):
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
                loss.backward()
                self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion
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
    return BaselinePrecisionMixedBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Precision Mixed: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
