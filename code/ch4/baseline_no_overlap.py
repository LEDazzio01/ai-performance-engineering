"""baseline_no_overlap.py - DDP without communication overlap (baseline).

DDP implementation without gradient overlap optimization.
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
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


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
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda:0")


class MultiLayerNet(nn.Module):
    """Multi-layer network for benchmarking."""
    
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BaselineNoOverlapBenchmark(Benchmark):
    """DDP without communication overlap - baseline."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self.rank = 0
        self.world_size = 1
        self.initialized = False
    
    def setup(self) -> None:
        """Setup: Initialize distributed environment and model."""
        setup_single_gpu_env()
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            self.initialized = True
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(0)
        
        torch.manual_seed(42)
        model = MultiLayerNet(1024).to(self.device)
        
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
        else:
            self.model = model
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        batch_size = 128
        self.data = torch.randn(batch_size, 1024, device=self.device)
        self.target = torch.randn(batch_size, 1, device=self.device)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: DDP training step without overlap."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_no_overlap", enable=enable_nvtx):
            output = self.model(self.data)
            loss = nn.functional.mse_loss(output, self.target)
            loss.backward()
            
            # Manual all-reduce without overlap (baseline)
            if self.world_size > 1:
                for param in self.model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad.mul_(1.0 / self.world_size)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if dist.is_initialized() and self.initialized:
            dist.destroy_process_group()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data tensor not initialized"
        if self.target is None:
            return "Target tensor not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != 128:
                    return f"Output batch size mismatch: expected 128, got {test_output.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineNoOverlapBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline No Overlap: {result.mean_ms:.3f} ms")
