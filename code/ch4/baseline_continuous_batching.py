"""baseline_continuous_batching.py - Baseline static batching in multi-GPU context.

Demonstrates static batching where batches are processed sequentially across GPUs.
No continuous batching: batches wait for full completion before next batch starts.
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
from ch4.gpu_requirements import skip_if_insufficient_gpus

from typing import Optional, List

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda")


class BaselineContinuousBatchingBenchmark(Benchmark):
    """Baseline: Static batching - batches processed sequentially across GPUs.
    
    Continuous batching: This baseline does not implement continuous batching.
    Batches are processed one at a time, with full synchronization between batches.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.batches = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self) -> None:
        """Setup: Initialize model and static batches."""
        skip_if_insufficient_gpus()

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
        
        # Simple model for multi-GPU inference
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device)
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.model.eval()
        
        # Baseline: Static batches - fixed batch sizes, processed sequentially
        # Each batch must complete fully before next batch starts
        batch_size = 4
        num_batches = 8
        self.batches = [
            torch.randn(batch_size, 32, 256, device=self.device)
            for _ in range(num_batches)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Static batching - sequential processing."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Process batches sequentially
                # Each batch waits for previous batch to complete
                # No continuous batching: cannot add/remove requests mid-batch
                for batch in self.batches:
                    # Forward pass
                    output = self.model(batch)
                    
                    # Synchronize across GPUs (if distributed)
                    if self.is_distributed:
                        dist.all_reduce(output, op=dist.ReduceOp.SUM)
                        output = output / self.world_size
                    
                    # Wait for completion before next batch
                    torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.batches = None
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
        if self.batches is None:
            return "Batches not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineContinuousBatchingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: continuous_batching")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
