"""baseline_dataparallel.py - DataParallel baseline (anti-pattern).

Uses DataParallel which has significant overhead due to GIL and single-threaded execution.
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
import torch.optim as optim


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


class SimpleNet(nn.Module):
    """Simple neural network for benchmarking."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class BaselineDataParallelBenchmark(Benchmark):
    """DataParallel baseline - has GIL overhead and single-threaded execution."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self.input_size = 1024
        self.hidden_size = 256
        self.batch_size = 512
    
    def setup(self) -> None:
        """Setup: Initialize model, optimizer, and data."""
        torch.manual_seed(42)
        
        # Keep input tensors on CPU so DataParallel copies every iteration.
        model = SimpleNet(self.input_size, self.hidden_size).to(self.device)
        self.model = nn.DataParallel(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        self.data = torch.randn(self.batch_size, self.input_size)
        self.target = torch.randn(self.batch_size, 1)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: DataParallel training step."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("dataparallel", enable=enable_nvtx):
            for _ in range(2):
                gpu_data = self.data.to(self.device, non_blocking=False)
                gpu_target = self.target.to(self.device, non_blocking=False)
                output = self.model(gpu_data)
                loss = nn.functional.mse_loss(output, gpu_target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
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
                if test_output.shape[0] != self.batch_size:
                    return f"Output batch size mismatch: expected {self.batch_size}, got {test_output.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineDataParallelBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline DataParallel: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
