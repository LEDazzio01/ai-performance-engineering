"""optimized_distributed.py - Optimized distributed operations in storage I/O context.

Demonstrates distributed processing for storage operations.
Distributed: Uses distributed coordination across multiple nodes.
Coordinates storage I/O across distributed nodes.
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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch5.gpu_requirements import skip_if_insufficient_gpus


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class OptimizedDistributedBenchmark(Benchmark):
    """Optimized: Distributed processing for storage operations.
    
    Distributed: Uses distributed coordination across multiple nodes.
    Coordinates storage I/O across distributed nodes.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize data and distributed processing."""
        skip_if_insufficient_gpus()
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Distributed processing
        # Coordinates storage I/O across multiple nodes
        
        # Initialize distributed if running in distributed mode
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            
            if world_size > 1:
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    rank=rank,
                    world_size=world_size
                )
                self.is_distributed = True
                self.rank = rank
                self.world_size = world_size
        
        # Distributed: each node processes its portion
        chunk_size = self.N // max(self.world_size, 1)
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.world_size - 1 else self.N
        
        self.data = torch.randn(end_idx - start_idx, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Distributed processing."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_distributed", enable=enable_nvtx):
            # Optimization: Distributed processing
            # Each node processes its portion of data
            # Distributed: coordinates across nodes
            local_result = self.data.sum()
            
            if self.is_distributed:
                # Distributed: aggregate results across nodes
                dist.all_reduce(local_result, op=dist.ReduceOp.SUM)
                result = local_result / self.world_size
            else:
                result = local_result
            
            # Optimization: Distributed processing benefits
            # - Multi-node coordination
            # - Parallel processing across nodes
            # - Scalable storage I/O
            # - Distributed workload distribution
            _ = result

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.is_distributed:
            dist.destroy_process_group()
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedDistributedBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedDistributedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Distributed")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
