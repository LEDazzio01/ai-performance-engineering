"""baseline continuous batching - Baseline static batching implementation. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineContinuousBatchingBenchmark(Benchmark):
    """Baseline: Static batching - fixed batch size, no dynamic composition."""

    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.requests = None

    def setup(self) -> None:
        """Setup: Initialize model and static batches."""
        torch.manual_seed(42)
        # Baseline: Static batching - fixed batch size
        # All requests must wait for full batch before processing
        
        # Baseline uses FP32 (slower) while optimized uses FP16 (faster) - legitimate optimization
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).to(torch.float32).eval()
        
        # Baseline: Static batches - fixed size, no dynamic composition
        # Requests are grouped into fixed-size batches
        batch_size = 8
        self.requests = [
            torch.randn(batch_size, 32, 256, device=self.device)
            for _ in range(4)  # 4 static batches
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Static batching."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_continuous_batching", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Static batching
                # Process fixed-size batches sequentially
                # No dynamic composition - must wait for full batch
                for batch in self.requests:
                    _ = self.model(batch)


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.requests = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineContinuousBatchingBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Continuous Batching (Static): {result.mean_ms:.3f} ms")
