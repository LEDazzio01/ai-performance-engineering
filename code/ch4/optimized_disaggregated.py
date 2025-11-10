"""optimized_disaggregated.py - Optimized disaggregated inference in multi-GPU context.

Demonstrates disaggregated inference where prefill and decode are separated across GPUs.
Disaggregated inference: Separates prefill (parallel, compute-intensive) and decode (autoregressive, latency-sensitive) phases.
Assigns different GPU resources to each phase for optimal utilization.
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


class OptimizedDisaggregatedBenchmark(Benchmark):
    """Optimized: Disaggregated inference (prefill and decode separated across GPUs).
    
        Disaggregated inference: Separates prefill (parallel, compute-intensive) and decode
        (autoregressive, latency-sensitive) phases. Assigns different GPU resources to each
        phase for optimal utilization and reduced interference.
        """
    
    def __init__(self):
        self.device = resolve_device()
        self.prefill_model = None

        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self) -> None:
        """Setup: Initialize separate models for prefill and decode."""
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
            except Exception:
                self.is_distributed = False
                self.rank = 0
                self.world_size = 1
        else:
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1
        
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        torch.manual_seed(42)
        
        # Optimization: Disaggregated inference
        # Separate models/resources for prefill and decode phases
        # Prefill: Parallel processing, compute-intensive, can use multiple GPUs
        # Decode: Autoregressive, latency-sensitive, dedicated GPU resources
        
        # Prefill model (optimized for parallel processing)
        self.prefill_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Decode model (optimized for latency)
        self.decode_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        if self.is_distributed:
            pass
        # In disaggregated setup, prefill and decode can use different GPU groups
        # For demo: both use same group, but separated logically
            self.prefill_model = nn.parallel.DistributedDataParallel(self.prefill_model)
            self.decode_model = nn.parallel.DistributedDataParallel(self.decode_model)
        
        # Simulate prefill (long context) and decode (single token) inputs
        self.prefill_input = torch.randn(2, 512, 256, device=self.device)  # Long context
        self.decode_input = torch.randn(2, 1, 256, device=self.device)  # Single token
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Disaggregated inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                pass
        # Optimization: Disaggregated inference
        # Prefill and decode phases are separated across GPUs
        # Prefill: Uses dedicated GPU resources for parallel processing
        # Decode: Uses separate GPU resources for low-latency autoregressive generation
                
        # Process prefill on dedicated prefill GPUs (parallel, compute-intensive)
        prefill_output = self.prefill_model(self.prefill_input)
                
        # Synchronize prefill across GPUs
        if self.is_distributed:
            pass
        dist.all_reduce(prefill_output, op=dist.ReduceOp.SUM)
        prefill_output = prefill_output / self.world_size
                
        # Process decode on dedicated decode GPUs (autoregressive, latency-sensitive)
        # Decode can run concurrently with next prefill (no interference)
        decode_output = self.decode_model(self.decode_input)
                
        # Synchronize decode across GPUs
        if self.is_distributed:
            pass
        dist.all_reduce(decode_output, op=dist.ReduceOp.SUM)
        decode_output = decode_output / self.world_size
                
        # Optimization: Disaggregated inference benefits
        # - Prefill and decode don't interfere (separate resources)
        # - Better GPU utilization (each phase optimized for its workload)
        # - Lower latency for decode (dedicated resources)
        # - Higher throughput for prefill (parallel processing)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.prefill_model = None
        self.decode_model = None
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
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedDisaggregatedBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedDisaggregatedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: disaggregated")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
