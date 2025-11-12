"""baseline_memory.py - Baseline standard GPU memory allocation.

Demonstrates standard GPU memory allocation without optimization.
Memory: This baseline uses standard cudaMalloc without memory optimization.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import warnings

import torch
import torch.nn as nn
from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

BATCH_SIZE = 512
INPUT_DIM = 2048
HIDDEN_DIM = 2048
REPETITIONS = 8


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class BaselineMemoryBenchmark(Benchmark):
    """Baseline: Standard GPU memory allocation."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        self.repetitions = REPETITIONS
        self.host_batches: list[torch.Tensor] = []
        self._prev_threads: Optional[int] = None
        self._prev_interop_threads: Optional[int] = None
        self._threads_overridden = False
        self._interop_overridden = False
    
    @staticmethod
    def _safe_set_thread_fn(setter, value: int, label: str, warn=True) -> bool:
        """Try to set a torch threading knob without aborting the benchmark."""
        try:
            setter(value)
            return True
        except RuntimeError as err:
            if warn:
                warnings.warn(f"Unable to set {label} (continuing with defaults): {err}")
            return False
    
    def setup(self) -> None:
        """Setup: Initialize model with standard memory allocation."""
        torch.manual_seed(42)
        self._prev_threads = torch.get_num_threads()
        self._prev_interop_threads = torch.get_num_interop_threads()
        self._threads_overridden = self._safe_set_thread_fn(
            torch.set_num_threads, 1, "num_threads"
        )
        self._interop_overridden = self._safe_set_thread_fn(
            torch.set_num_interop_threads, 1, "num_interop_threads"
        )
        # Baseline: Standard GPU memory allocation
        # Memory optimization techniques include custom allocators, memory pooling, etc.
        # This baseline uses standard PyTorch memory allocation
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, self.input_dim),
        ).to(self.device).eval()
        self.host_batches = [
            torch.randint(
                0,
                256,
                (self.batch_size, self.input_dim),
                device="cpu",
                dtype=torch.uint8,
            )
            for _ in range(self.repetitions)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard memory allocation."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Baseline: Standard memory allocation and usage
        # No memory optimization - uses default PyTorch allocator
        with nvtx_range("baseline_memory", enable=enable_nvtx):
            with torch.no_grad():
                # Naive path: CPU transforms + synchronous host->device transfer.
                for compressed in self.host_batches:
                    host_batch = compressed.to(dtype=torch.float32)
                    host_batch.mul_(1.0 / 255.0)
                    host_batch.add_(-0.5)
                    host_batch.mul_(2.0)
                    host_batch.tanh_()
                    device_batch = host_batch.to(self.device, non_blocking=False)
                    _ = self.model(device_batch)
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.host_batches = []
        if self._prev_threads is not None and self._threads_overridden:
            self._safe_set_thread_fn(
                torch.set_num_threads, self._prev_threads, "num_threads reset", warn=False
            )
        if self._prev_interop_threads is not None and self._interop_overridden:
            self._safe_set_thread_fn(
                torch.set_num_interop_threads,
                self._prev_interop_threads,
                "num_interop_threads reset",
                warn=False,
            )
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
