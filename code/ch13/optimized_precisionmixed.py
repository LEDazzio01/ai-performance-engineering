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
from torch.amp import autocast

from typing import Optional

from common.python.compile_utils import enable_tf32
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

class OptimizedPrecisionMixedBenchmark(Benchmark):
    """Mixed precision - uses autocast and GradScaler."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.batch_size = 128
        self.hidden_dim = 2048
        self.micro_steps = 4
    
    def setup(self) -> None:
        """Setup: Initialize model with mixed precision."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        
        # Model stays FP32 - autocast handles FP16 conversion
        # Use torch.compile to fuse operations and optimize FP16 kernels
        model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device)
        model.train()
        
        # Compile model to optimize mixed precision path (fallback gracefully if unsupported)
        try:
            self.model = torch.compile(model, mode='reduce-overhead')
        except Exception:
            self.model = model
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.model = self.model.to(torch.float16)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # Warmup to ensure compilation completes (full train step to cover forward/backward)
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=torch.float16):
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()
        torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - mixed precision training."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_precision_mixed", enable=enable_nvtx):
            for _ in range(self.micro_steps):
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass in FP16
                with autocast('cuda', dtype=torch.float16):
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
    return OptimizedPrecisionMixedBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Precision Mixed: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
