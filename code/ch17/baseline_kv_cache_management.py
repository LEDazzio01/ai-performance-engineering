"""baseline_kv_cache_management.py - Baseline KV cache without management in inference/profiling.

Demonstrates KV cache usage without proper management.
KV cache management: This baseline does not implement efficient KV cache management.
Recomputes keys/values each step instead of reusing cache.
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
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class BaselineKVCacheManagementBenchmark(Benchmark):
    """Baseline: KV cache without management (recomputes each step).
    
    KV cache management: This baseline does not implement efficient KV cache management.
    Recomputes keys/values each step instead of reusing cached values.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
    
    def setup(self) -> None:
        """Setup: Initialize model without KV cache management."""
        torch.manual_seed(42)
        # Baseline: No KV cache management - recomputes each step
        # KV cache management reuses cached keys/values for efficiency
        # This baseline does not reuse cache
        
        hidden_dim = 256
        num_heads = 8
        
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device).eval()
        
        # Simulate autoregressive generation
        batch_size = 4
        self.inputs = [
            torch.randn(batch_size, 1, hidden_dim, device=self.device)
            for _ in range(32)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: KV cache without management."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_kv_cache_management", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: No KV cache management
                # Recomputes keys/values for all previous tokens each step
                # Inefficient: wastes computation on repeated work
                
                for step, query in enumerate(self.inputs):
                    # Recompute K, V for all tokens up to this point
                    # No cache reuse - inefficient KV cache management
                    all_inputs = torch.cat(self.inputs[:step+1], dim=1)
                    output, _ = self.model(query, all_inputs, all_inputs)
                    
                    # Baseline: No KV cache management
                    # Each step recomputes everything (inefficient)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
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
        if self.inputs is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineKVCacheManagementBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineKVCacheManagementBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: kv_cache_management")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
