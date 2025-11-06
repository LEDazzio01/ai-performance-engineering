"""baseline_paged_attention.py - Baseline attention without paged attention in MoE context.

Demonstrates attention computation without paged attention optimization.
Paged attention: This baseline does not use paged attention for KV cache management.
Uses contiguous memory allocation, causing fragmentation and inefficiency.
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
import torch.nn.functional as F

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class BaselinePagedAttentionBenchmark(Benchmark):
    """Baseline: Attention without paged attention.
    
    Paged attention: This baseline does not use paged attention for KV cache management.
    Uses contiguous memory allocation, causing fragmentation and inefficient memory usage.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.kv_cache = None
        self.inputs = None
    
    def setup(self) -> None:
        """Setup: Initialize model and contiguous KV cache."""
        torch.manual_seed(42)
        # Baseline: No paged attention - contiguous memory allocation
        # Paged attention uses non-contiguous pages for efficient memory management
        # This baseline allocates contiguous memory for each sequence
        
        hidden_dim = 512
        num_heads = 8
        head_dim = hidden_dim // num_heads
        
        # Simple attention model
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device).eval()
        
        # Baseline: Contiguous KV cache allocation (no paging)
        # Each sequence gets full contiguous memory allocation
        batch_size = 4
        max_seq_len = 256
        self.kv_cache = {
            'k': torch.zeros(batch_size, max_seq_len, num_heads, head_dim, device=self.device),
            'v': torch.zeros(batch_size, max_seq_len, num_heads, head_dim, device=self.device),
        }
        
        # Simulate autoregressive generation
        self.inputs = [
            torch.randn(batch_size, 1, hidden_dim, device=self.device)
            for _ in range(64)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention without paged attention."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Contiguous KV cache (no paging)
                # Memory is allocated contiguously, causing fragmentation
                # Cannot efficiently handle variable-length sequences
                
                for step, query in enumerate(self.inputs):
                    # Compute new K, V
                    _, k_new, v_new = self.model(query, query, query, need_weights=False)
                    
                    # Store in contiguous cache (inefficient for variable lengths)
                    self.kv_cache['k'][:, step:step+1, :, :] = k_new
                    self.kv_cache['v'][:, step:step+1, :, :] = v_new
                    
                    # Attention computation using all cached K, V
                    k_all = self.kv_cache['k'][:, :step+1, :, :]
                    v_all = self.kv_cache['v'][:, :step+1, :, :]
                    
                    # Baseline: No paged attention
                    # Contiguous allocation causes memory waste for variable-length sequences

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.kv_cache = None
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
        if self.kv_cache is None:
            return "KV cache not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselinePagedAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselinePagedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: paged_attention")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
