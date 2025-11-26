"""baseline_nccl.py - Baseline without NCCL in distributed training context."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineNcclBenchmark(BaseBenchmark):
    """Baseline: No NCCL - CPU-based or inefficient communication.
    
    NCCL: This baseline does not use NCCL for collective communication.
    Uses CPU-based or inefficient communication patterns.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input = None
        self.output = None
        self.batch = 256
        self.hidden = 2048
        tokens = self.batch * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model without NCCL."""
        torch.manual_seed(42)
        # Baseline: No NCCL - CPU-based communication
        # NCCL provides optimized GPU-to-GPU collective communication
        # This baseline does not use NCCL
        
        self.model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
        ).to(self.device).eval()
        
        self.input = torch.randn(256, 2048, device=self.device)
        self.output = torch.zeros_like(self.input)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without NCCL."""
        with self._nvtx_range("baseline_nccl"):
            with torch.no_grad():
                output = self.model(self.input)
                
                # Naively copy shards to CPU to aggregate
                shards = torch.chunk(output, chunks=4, dim=0)
                reduced = sum(shard.cpu() for shard in shards) / 4.0
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
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
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


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineNcclBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: nccl")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
