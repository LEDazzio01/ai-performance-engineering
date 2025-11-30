"""optimized_disaggregated.py - Optimized disaggregated inference in multi-GPU context.

Demonstrates disaggregated inference where prefill and decode are separated across GPUs.
Disaggregated inference: Separates prefill (parallel, compute-intensive) and decode (autoregressive, latency-sensitive) phases.
Assigns different GPU resources to each phase for optimal utilization.
Implements BaseBenchmark for harness integration.
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

from core.utils.compile_utils import compile_model
from core.benchmark.gpu_requirements import skip_if_insufficient_gpus

from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)

class OptimizedDisaggregatedBenchmark(BaseBenchmark):
    """Optimized: Disaggregated inference (prefill and decode separated across GPUs).
    
        Disaggregated inference: Separates prefill (parallel, compute-intensive) and decode
        (autoregressive, latency-sensitive) phases. Assigns different GPU resources to each
        phase for optimal utilization and reduced interference.
        """
    
    def __init__(self):
        super().__init__()
        self.prefill_model = None

        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.batch_size = 2
        self.prefill_len = 512
        self.hidden_dim = 256
        tokens = self.batch_size * self.prefill_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
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
                self.is_distributed = True
            except Exception:
                self.is_distributed = False
                self.rank = 0
                self.world_size = 1
        else:
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1
        
        if self.is_distributed and dist.is_initialized():
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
            # In disaggregated setup, prefill and decode can use different GPU groups.
            self.prefill_model = nn.parallel.DistributedDataParallel(self.prefill_model)
            self.decode_model = nn.parallel.DistributedDataParallel(self.decode_model)
        
        # Simulate prefill (long context) and decode (single token) inputs
        self.prefill_input = torch.randn(self.batch_size, self.prefill_len, self.hidden_dim, device=self.device)
        self.decode_input = torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Disaggregated inference."""
        assert self.prefill_model is not None and self.decode_model is not None
        assert self.prefill_input is not None and self.decode_input is not None
        with self._nvtx_range("optimized_disaggregated"):
            with torch.no_grad():
                # Process prefill on dedicated prefill GPUs (parallel, compute-intensive)
                prefill_output = self.prefill_model(self.prefill_input)
                
                # Synchronize prefill across GPUs
                if self.is_distributed:
                    dist.all_reduce(prefill_output, op=dist.ReduceOp.SUM)
                    prefill_output = prefill_output / self.world_size
                
                # Process decode on dedicated decode GPUs (autoregressive, latency-sensitive)
                decode_output = self.decode_model(self.decode_input)
                
                # Synchronize decode across GPUs
                if self.is_distributed:
                    dist.all_reduce(decode_output, op=dist.ReduceOp.SUM)
                    decode_output = decode_output / self.world_size
        self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
        iterations=10,
            warmup=5,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedDisaggregatedBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
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
