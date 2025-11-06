"""optimized_continuous_batching.py - Optimized continuous batching in multi-GPU context.

Demonstrates continuous batching where requests can be added/removed dynamically.
Batches are processed with variable sizes, allowing better GPU utilization.
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
import torch.distributed as dist

from common.python.compile_utils import compile_model

from typing import Optional, List, Deque
from collections import deque

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda")

class OptimizedContinuousBatchingBenchmark:
    """Optimized: Continuous batching - dynamic batch composition.
    
        Continuous batching: Implements continuous batching where requests can be
        added or removed dynamically. Batches are composed from a request queue,
        allowing better GPU utilization by filling batches optimally.
        """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None

        self.request_queue = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self) -> None:
        """Setup: Initialize model and request queue."""
        
        # Initialize distributed if available
        if dist.is_available() and torch.cuda.device_count() > 1:
            try:
                if not dist.is_initialized():
                    import os
                    if 'MASTER_ADDR' not in os.environ:
                        os.environ['MASTER_ADDR'] = 'localhost'
                    if 'MASTER_PORT' not in os.environ:
                        os.environ['MASTER_PORT'] = '12355'
                    if 'RANK' not in os.environ:
                        os.environ['RANK'] = '0'
                    if 'WORLD_SIZE' not in os.environ:
                        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
                    dist.init_process_group(backend='nccl', init_method='env://')
                    self.is_distributed = True
                    self.rank = dist.get_rank()
                    self.world_size = dist.get_world_size()
            except Exception:
                self.is_distributed = False
                self.rank = 0
                self.world_size = 1
        
        # Simple model for multi-GPU inference
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device)
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.model.eval()
        
        # Optimization: Continuous batching - dynamic request queue
        # Requests can be added/removed as they complete
        # Simulate incoming requests with varying sizes
        self.request_queue = deque([
        torch.randn(2, 32, 256, device=self.device),  # Small request
        torch.randn(4, 32, 256, device=self.device),  # Medium request
        torch.randn(3, 32, 256, device=self.device),  # Medium request
        torch.randn(5, 32, 256, device=self.device),  # Large request
        torch.randn(1, 32, 256, device=self.device),  # Small request
        torch.randn(4, 32, 256, device=self.device),  # Medium request
        torch.randn(2, 32, 256, device=self.device),  # Small request
        torch.randn(3, 32, 256, device=self.device),  # Medium request
        ])
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Continuous batching - dynamic batch composition."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                pass
        # Optimization: Continuous batching
        # Compose batches dynamically from request queue
        # Can combine multiple requests into a single batch
        max_batch_size = 8
        processed_requests = []
                
        while self.request_queue:
            # Compose batch from queue (continuous batching)
            current_batch = []
            current_size = 0
                    
            # Add requests to batch until max size reached
            while self.request_queue and current_size < max_batch_size:
                request = self.request_queue.popleft()
                request_size = request.shape[0]
                        
                if current_size + request_size <= max_batch_size:
                    current_batch.append(request)
                    current_size += request_size
                else:
                    # Too large, put back
                    self.request_queue.appendleft(request)
                    break
                    
            if current_batch:
                # Concatenate requests into single batch
                batch = torch.cat(current_batch, dim=0)
                        
                # Forward pass
                output = self.model(batch)
                        
                # Synchronize across GPUs (if distributed)
                if self.is_distributed:
                    dist.all_reduce(output, op=dist.ReduceOp.SUM)
                    output = output / self.world_size
                        
                processed_requests.append(len(current_batch))
                    
            # Continuous batching: New requests can be added to queue
            # without waiting for full batch completion
            # This allows better GPU utilization
                
            # Simulate new requests arriving (continuous batching benefit)
            # In real system, requests arrive asynchronously

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.request_queue = None
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
        iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.request_queue is None:
            return "Request queue not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedContinuousBatchingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: continuous_batching")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
