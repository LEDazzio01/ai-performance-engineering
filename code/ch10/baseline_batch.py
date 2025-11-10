"""baseline_batch.py - Baseline small batch size in GEMM context.

Demonstrates operations with small batch size, limiting GPU utilization.
Batch: This baseline uses small batch size.
Small batches do not fully utilize GPU resources.
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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch10.workload_config import WORKLOAD, is_smoke_test


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineBatchBenchmark(Benchmark):
    """Baseline: Small batch size - limited GPU utilization.
    
    Batch: This baseline uses small batch size.
    Small batches do not fully utilize GPU resources.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model: nn.Sequential | None = None
        self.inputs: torch.Tensor | None = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.micro_batch_size = self.workload.baseline_micro_batch_for_mode(self.smoke_test)
        self.micro_batches = self.workload.baseline_micro_batches_for_mode(self.smoke_test)
        self.hidden_dim = self.workload.hidden_dim
        self.ffn_dim = self.workload.ffn_dim
        self._last_total = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model with small batch size."""
        torch.manual_seed(42)
        # Baseline: Small batch size - limited GPU utilization
        # Batch size optimization improves GPU utilization
        # This baseline uses small batch size
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        # Small batch size processed sequentially (inefficient)
        self.inputs = torch.randn(
            self.micro_batches,
            self.micro_batch_size,
            self.hidden_dim,
            device=self.device,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with small batch size."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("batch", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Small batch size
                # Small batches do not fully utilize GPU resources
                # Batch size optimization would improve utilization
                if self.model is None or self.inputs is None:
                    raise RuntimeError("Benchmark not configured")
                
                # Process micro-batches sequentially, forcing repeated launches.
                total = 0.0
                for idx in range(self.micro_batches):
                    output = self.model(self.inputs[idx])
                    total += float(output.sum())
                # Stash total to keep the reduction alive without synchronization bugs.
                self._last_total = total
                torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
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
        if self.inputs is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineBatchBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineBatchBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Batch")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
