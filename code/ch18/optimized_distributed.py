"""optimized_distributed.py - Optimized distributed operations in FlexAttention/KV cache context.

Demonstrates distributed processing for attention operations.
Distributed: Uses distributed coordination across multiple nodes.
Coordinates attention computation across distributed nodes.
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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")

class OptimizedDistributedBenchmark(Benchmark):
    """Optimized: Distributed processing for attention operations.
    
    Distributed: Uses distributed coordination across multiple nodes.
    Coordinates attention computation across distributed nodes.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

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
        # Coordinates attention computation across multiple nodes
        
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
        
        hidden_dim = 256
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        ).to(self.device).eval()
        
        # Distributed: each node processes its portion
        batch_size = 4
        seq_len = 128
        chunk_size = seq_len // max(self.world_size, 1)
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.world_size - 1 else seq_len
        
        self.input = torch.randn(batch_size, end_idx - start_idx, hidden_dim, device=self.device)
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
                # Each node processes its portion of attention
                # Distributed: coordinates across nodes
                output, _ = self.model(self.input, self.input, self.input)
                
                if self.is_distributed:
                    # Distributed: aggregate results across nodes
                    dist.all_reduce(output, op=dist.ReduceOp.SUM)
                    output = output / self.world_size
                
                # Optimization: Distributed processing benefits
                # - Multi-node coordination
                # - Parallel processing across nodes
                # - Scalable attention computation
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
