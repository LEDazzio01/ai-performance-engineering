"""baseline_nccl.py - Baseline: CPU-based reduction (inefficient).

This baseline copies each shard to CPU for aggregation, then back to GPU.
This is visibly slower than the optimized path which stays on GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineNcclBenchmark(BaseBenchmark):
    """Baseline: CPU-based reduction - copies each shard to CPU for aggregation."""
    
    def __init__(self):
        super().__init__()
        self.skip_output_check = True
        # Match optimized workload size
        self.batch_size = 1024
        self.hidden_dim = 4096
        self.inner_dim = 8192
        self.num_shards = 8  # More shards = more CPU round-trips
        
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def skip_output_verification(self) -> bool:
        return True
    
    def setup(self) -> None:
        """Setup: Initialize model."""
        torch.manual_seed(42)
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        self.input = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        shard_size = self.batch_size // self.num_shards
        self.output = torch.zeros(shard_size, self.hidden_dim, device=self.device)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: CPU-based reduction (inefficient)."""
        with self._nvtx_range("baseline_nccl"):
            with torch.no_grad():
                output = self.model(self.input)
                
                # Naively copy each shard to CPU to aggregate (slow!)
                shards = torch.chunk(output, chunks=self.num_shards, dim=0)
                # This is the bottleneck: multiple GPU->CPU copies + CPU addition
                cpu_shards = [shard.cpu() for shard in shards]
                reduced = sum(cpu_shards) / float(self.num_shards)
                # Copy result back to GPU
                self.output = reduced.to(self.device)
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
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
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineNcclBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
