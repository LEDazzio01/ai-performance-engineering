"""baseline_disaggregated.py - Baseline monolithic inference in multi-GPU context."""

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

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineDisaggregatedBenchmark(BaseBenchmark):
    """Baseline: Monolithic inference (prefill and decode share resources across GPUs).
    
    Disaggregated inference: This baseline does not separate prefill and decode phases.
    Both phases compete for same GPU resources, causing interference and poor utilization.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.prefill_input = None
        self.decode_input = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.batch_size = 2
        self.prefill_len = 512
        self.hidden_dim = 256
        tokens = self.batch_size * (self.prefill_len + 1)  # include decode token
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and inputs."""
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
        self.prefill_input = torch.randn(self.batch_size, self.prefill_len, self.hidden_dim, device=self.device)
        self.decode_input = torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Monolithic inference."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

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
        self._synchronize()
                
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
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
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
        if self.model is None:
            return "Model not initialized"
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineDisaggregatedBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineDisaggregatedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: disaggregated")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
