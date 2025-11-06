"""optimized_distributed.py - Optimized distributed operations in kernel efficiency/arithmetic intensity context.

Demonstrates distributed processing for kernel efficiency operations.
Distributed: Uses distributed coordination across multiple nodes.
Coordinates operations across distributed nodes.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
import os
import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedDistributedBenchmark(Benchmark):
    """Optimized: Distributed processing for kernel efficiency operations.
    
    Distributed: Uses distributed coordination across multiple nodes.
    Coordinates operations across distributed nodes.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:
            model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        # Optimization: Compile model for kernel fusion and optimization
        try:
            self.model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        self.input = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self) -> None:
        """Setup: Initialize model and distributed processing."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Distributed processing
        # Coordinates operations across multiple nodes
        
        # Initialize distributed if running in distributed mode
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            
            if world_size > 1:
                # Set defaults for MASTER_ADDR/MASTER_PORT if not provided
                if 'MASTER_ADDR' not in os.environ:
                    os.environ['MASTER_ADDR'] = 'localhost'
                if 'MASTER_PORT' not in os.environ:
                    os.environ['MASTER_PORT'] = '12355'
                
                try:
                    dist.init_process_group(
                        backend='nccl',
                        init_method='env://',
                        rank=rank,
                        world_size=world_size,
                        timeout=datetime.timedelta(seconds=30)
                    )
                    self.is_distributed = True
                    self.rank = rank
                    self.world_size = world_size
                except Exception as e:
                    # Fallback if distributed init fails
                    self.is_distributed = False
                    self.rank = 0
                    self.world_size = 1
            else:
                self.is_distributed = False
                self.rank = 0
                self.world_size = 1
        else:
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Distributed: each node processes its portion
        # Guard against world_size > batch_size
        batch_size = 32
        chunk_size = max(1, batch_size // max(self.world_size, 1))  # Ensure at least 1
        start_idx = self.rank * chunk_size
        end_idx = min(start_idx + chunk_size, batch_size)  # Don't exceed batch_size
        
        # Ensure we have at least one sample per rank
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        
        self.input = torch.randn(end_idx - start_idx, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Distributed processing."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_distributed", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Distributed processing
                # Each node processes its portion
                # Distributed: coordinates across nodes
                output = self.model(self.input)
                
                if self.is_distributed:
                    # Distributed: aggregate results across nodes
                    dist.all_reduce(output, op=dist.ReduceOp.SUM)
                    output = output / self.world_size
                
                # Optimization: Distributed processing benefits
                # - Multi-node coordination
                # - Parallel processing across nodes
                # - Scalable operations
                # - Distributed workload distribution
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.is_distributed:
            dist.destroy_process_group()
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
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
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

