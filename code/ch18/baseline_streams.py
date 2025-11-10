"""baseline_streams.py - Baseline without CUDA streams in FlexAttention/KV cache context.

Demonstrates operations without CUDA streams for parallel execution.
Streams: This baseline does not use CUDA streams.
Operations execute sequentially without overlap.
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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class BaselineStreamsBenchmark(Benchmark):
    """Baseline: Sequential execution without CUDA streams.
    
    Streams: This baseline does not use CUDA streams.
    Operations execute sequentially without overlap.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input1 = None
        self.input2 = None
        self.batch = 8
        self.seq_len = 256
        self.hidden_dim = 512
    
    def setup(self) -> None:
        """Setup: Initialize model without streams."""
        torch.manual_seed(42)
        # Baseline: No CUDA streams - sequential execution
        # CUDA streams allow parallel execution of independent operations
        # This baseline does not use streams
        
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True
        ).to(self.device).eval()
        
        self.input1 = torch.randn(self.batch, self.seq_len, self.hidden_dim, device=self.device)
        self.input2 = torch.randn(self.batch, self.seq_len, self.hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential execution without streams."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_streams", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Sequential execution (no streams)
                # Sequential execution: launch two attention passes back-to-back
                output1, _ = self.model(self.input1, self.input1, self.input1)
                torch.cuda.synchronize()
                output2, _ = self.model(self.input2, self.input2, self.input2)
                torch.cuda.synchronize()
                _ = output1 + output2

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input1 = None
        self.input2 = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input1 is None or self.input2 is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineStreamsBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineStreamsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Streams")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
