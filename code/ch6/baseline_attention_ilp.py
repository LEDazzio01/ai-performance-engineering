"""baseline_attention_ilp.py - Baseline attention with low ILP.

Demonstrates attention operations that limit instruction-level parallelism.
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
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class BaselineAttentionILPBenchmark(Benchmark):
    """Baseline: Attention with low ILP (sequential operations)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize attention model."""
        torch.manual_seed(42)
        # Baseline: Attention with low ILP
        # Sequential attention operations limit instruction-level parallelism
        # This baseline does not optimize for ILP
        self.model = nn.MultiheadAttention(256, 8, batch_first=True)
        self.model = self.model.to(self.device).eval()
        self.input = torch.randn(4, 32, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention with low ILP."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_attention_ilp", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Sequential attention operations (low ILP)
                # Attention operations performed sequentially limit ILP
                # This baseline does not optimize for instruction-level parallelism
                # Attention mechanism processes queries, keys, values sequentially
                _ = self.model(self.input, self.input, self.input)[0]
                # Low ILP: Sequential attention operations don't expose parallelism

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
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
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineAttentionILPBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Attention ILP: {result.mean_ms:.3f} ms")
    print("  Note: Sequential attention operations limit instruction-level parallelism")
