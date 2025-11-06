"""baseline_reinit_comm.py - Reinitializing NCCL every iteration (baseline).

Anti-pattern: reinitializing NCCL communicator on every iteration.
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
    return torch.device("cuda")


class BaselineReinitCommBenchmark(Benchmark):
    """Reinitializing NCCL every iteration - poor pattern."""
    
    def __init__(self):
        self.device = resolve_device()
        self.rank = 0
        self.world_size = 1
        self.tensor = None
        self.initialized = False
    
    def setup(self) -> None:
        """Setup: Configure distributed environment."""
        setup_single_gpu_env()
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(0)
        self.tensor = torch.ones(1, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Reinitialize NCCL every iteration."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_reinit_comm", enable=enable_nvtx):
            # Anti-pattern: reinitialize NCCL every iteration
            if dist.is_initialized():
                dist.destroy_process_group()
            
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )
            
            # Perform all-reduce
            dist.all_reduce(self.tensor)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        self.tensor = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        if not dist.is_initialized():
            return "Distributed process group not initialized"
        # Check tensor shape and values
        if self.tensor.shape != (1,):
            return f"Tensor shape mismatch: expected (1,), got {self.tensor.shape}"
        if not torch.isfinite(self.tensor).all():
            return "Tensor contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineReinitCommBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Reinit Comm: {result.mean_ms:.3f} ms")
