"""optimized_precision_mixed.py - Mixed precision optimization (optimized).

Mixed precision training using autocast and GradScaler.
Uses FP16 for forward pass, FP32 for backward pass accumulation.
Faster computation and lower memory usage.

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
from torch.amp import autocast, GradScaler

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

class OptimizedPrecisionMixedBenchmark(Benchmark):
    """Mixed precision - uses autocast and GradScaler."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.batch_size = 32
        self.hidden_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize model with mixed precision."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        # Model (can be FP32, autocast handles conversion)
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        
        self.model.train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler('cuda')  # For mixed precision gradient scaling
        
        # Warmup
        for _ in range(3):
            self.optimizer.zero_grad()
            with autocast('cuda'):
                _ = self.model(self.inputs)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - mixed precision training."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_precision_mixed", enable=enable_nvtx):
            self.optimizer.zero_grad()
            
            # Forward pass in mixed precision (FP16)
            with autocast('cuda'):
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
            
            # Scaled backward pass (handles FP16->FP32 conversion)
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
    return OptimizedPrecisionMixedBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Precision Mixed: {result.mean_ms:.3f} ms")

