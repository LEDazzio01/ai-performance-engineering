"""optimized_training_single.py - Optimized training loop (optimized).

Uses fused AdamW, Ampere mixed precision, and compilation to simulate
high-throughput training even on a single GPU.

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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")

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

class OptimizedTrainingDistributedBenchmark(Benchmark):
    """Optimized training loop leveraging AMP, fused optimizers, and compilation."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.batch_size = 8
        self.hidden_dim = 4096
    
    def setup(self) -> None:
        """Setup: Initialize optimized single-GPU training stack."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        base_model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.model = torch.compile(base_model, mode="reduce-overhead")
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, fused=True)
        except TypeError:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - optimized training."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("training", enable=enable_nvtx):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion, self.scaler
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
    return OptimizedTrainingDistributedBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    timing = result.timing
    if timing:
        print(f"\nOptimized Distributed Training: {timing.mean_ms:.3f} ms")
    else:
        print("No timing data available")
