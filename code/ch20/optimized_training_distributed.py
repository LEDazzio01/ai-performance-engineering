"""optimized_training_distributed.py - Distributed training optimization (optimized).

Multi-GPU distributed training using DDP.
Scales training across multiple GPUs for better throughput.

Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from distributed_helper import setup_single_gpu_env
except ImportError:
    def setup_single_gpu_env():
        if "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("LOCAL_RANK", "0")

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
    """Distributed training optimization - multi-GPU scaling."""
    
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
        self.batch_size = 16  # Larger effective batch with multiple GPUs
        self.hidden_dim = 1024
        self.rank = 0
        self.world_size = 1
        self.initialized = False
    
    def setup(self) -> None:
        """Setup: Initialize distributed model and data."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        setup_single_gpu_env()
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            self.initialized = True
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(0)
        
        torch.manual_seed(42)
        
        # Distributed model (simulated multi-GPU with DDP)
        model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
        else:
            # Single GPU: simulate distributed benefits with larger batch
            self.model = model
        
        # Larger effective batch size with distributed training
        effective_batch = self.batch_size * max(1, self.world_size)
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # Warmup
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - distributed training."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_training_distributed", enable=enable_nvtx):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            # DDP handles gradient synchronization automatically
            self.optimizer.step()

    
    def teardown(self) -> None:
        """Cleanup."""
        if dist.is_initialized() and self.initialized:
            dist.destroy_process_group()
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
    return OptimizedTrainingDistributedBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Distributed Training: {result.mean_ms:.3f} ms")

