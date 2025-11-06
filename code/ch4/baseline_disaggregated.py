"""baseline_disaggregated.py - Baseline monolithic inference in multi-GPU context.

Demonstrates monolithic inference where prefill and decode share same resources across GPUs.
Disaggregated inference: This baseline does not separate prefill and decode phases.
Prefill and decode compete for same GPU resources, causing interference.
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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda")


class BaselineDisaggregatedBenchmark(Benchmark):
    """Baseline: Monolithic inference (prefill and decode share resources across GPUs).
    
    Disaggregated inference: This baseline does not separate prefill and decode phases.
    Both phases compete for same GPU resources, causing interference and poor utilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.prefill_input = None
        self.decode_input = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self) -> None:
        """Setup: Initialize model and inputs."""
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
        
        torch.manual_seed(42)
        # Baseline: Monolithic inference - prefill and decode share same resources
        # Disaggregated inference separates prefill (parallel) and decode (autoregressive)
        # This baseline does not separate prefill and decode phases
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.model = self.model.to(self.device).eval()
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        # Simulate prefill (long context) and decode (single token) inputs
        self.prefill_input = torch.randn(2, 512, 256, device=self.device)  # Long context
        self.decode_input = torch.randn(2, 1, 256, device=self.device)  # Single token
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Monolithic inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Monolithic inference
                # Prefill and decode phases share same resources across GPUs
                # This causes interference - prefill blocks decode and vice versa
                # Disaggregated inference separates these phases for better efficiency
                
                # Process prefill (long context) - competes with decode for resources
                prefill_output = self.model(self.prefill_input)
                
                # Synchronize across GPUs
                if self.is_distributed:
                    dist.all_reduce(prefill_output, op=dist.ReduceOp.SUM)
                    prefill_output = prefill_output / self.world_size
                
                # Process decode (autoregressive) - competes with prefill for resources
                decode_output = self.model(self.decode_input)
                
                # Synchronize across GPUs
                if self.is_distributed:
                    dist.all_reduce(decode_output, op=dist.ReduceOp.SUM)
                    decode_output = decode_output / self.world_size
                
                # Baseline: No separation - both phases interfere with each other
                # This leads to poor GPU utilization and latency spikes

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.prefill_input = None
        self.decode_input = None
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
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineDisaggregatedBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineDisaggregatedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: disaggregated")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
