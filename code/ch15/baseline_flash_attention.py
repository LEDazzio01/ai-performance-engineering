"""baseline_flash_attention.py - Baseline attention without FlashAttention in disaggregated inference.

Demonstrates standard attention computation without FlashAttention optimization.
Flash attention: This baseline does not use FlashAttention.
Uses standard attention with quadratic memory complexity.
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
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


class BaselineFlashAttentionBenchmark(Benchmark):
    """Baseline: Standard attention without FlashAttention.
    
    Flash attention: This baseline does not use FlashAttention.
    Uses standard attention with quadratic memory complexity.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize attention model without FlashAttention."""
        torch.manual_seed(42)
        # Baseline: Standard attention - no FlashAttention optimization
        # FlashAttention reduces memory complexity from O(seq_len^2) to O(seq_len)
        # This baseline does not use FlashAttention
        
        hidden_dim = 256
        num_heads = 8
        
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device).eval()
        
        # Input sequence
        batch_size = 4
        seq_len = 512  # Long sequence to show FlashAttention benefit
        self.input = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard attention without FlashAttention."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_flash_attention", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Standard attention (no FlashAttention)
                # Computes full attention matrix: O(seq_len^2) memory
                # FlashAttention would use tiling to reduce memory
                output, _ = self.model(self.input, self.input, self.input)
                
                # Baseline: No FlashAttention benefits
                # Quadratic memory complexity for long sequences

    
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
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineFlashAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: flash_attention")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
