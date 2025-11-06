"""optimized_nccl.py - Optimized NCCL for multi-GPU communication in storage I/O context.

Demonstrates NCCL for efficient multi-GPU collective communication.
NCCL: Uses NCCL for optimized GPU-to-GPU communication.
Provides efficient allreduce, broadcast, and other collective operations.
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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class OptimizedNcclBenchmark(Benchmark):
    """Optimized: NCCL for efficient multi-GPU communication.
    
    NCCL: Uses NCCL for optimized GPU-to-GPU communication.
    Provides efficient allreduce, broadcast, and other collective operations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize data and NCCL communication."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: NCCL for multi-GPU communication
        # NCCL provides optimized GPU-to-GPU collective communication
        
        # Initialize NCCL if running in distributed mode
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
        
        self.data = torch.randn(self.N, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: NCCL collective communication."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_nccl", enable=enable_nvtx):
            # Optimization: NCCL for multi-GPU communication
            # Uses NCCL collective operations for efficient communication
            result = self.data.sum()
            
            if self.is_distributed:
                # NCCL: Allreduce for multi-GPU aggregation
                dist.all_reduce(result, op=dist.ReduceOp.SUM)
                result = result / self.world_size
                
                # NCCL: Broadcast for multi-GPU synchronization
                dist.broadcast(result, src=0)
            
            # Optimization: NCCL benefits
            # - Efficient GPU-to-GPU communication
            # - Optimized collective operations (allreduce, broadcast)
            # - Better performance than CPU-based communication
            # - Hardware-optimized communication patterns
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
    return OptimizedNcclBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedNcclBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Nccl")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
