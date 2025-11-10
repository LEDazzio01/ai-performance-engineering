"""baseline_paged_attention.py - Baseline attention without paged attention in inference/profiling.

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


class BaselinePagedAttentionBenchmark(Benchmark):
    """Baseline: Attention without paged attention."""

    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.k_cache = None
        self.v_cache = None
        self.inputs = None
        self.batch_size = 2
        self.hidden_dim = 256
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.max_seq_len = 1024
        self.steps = 256

    def setup(self) -> None:
        """Setup: Initialize model and contiguous KV cache."""
        torch.manual_seed(42)
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        ).to(self.device).eval()

        dtype = self.model.in_proj_weight.dtype
        shape = (self.batch_size, self.max_seq_len, self.num_heads, self.head_dim)
        self.k_cache = torch.zeros(shape, device=self.device, dtype=dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device, dtype=dtype)
            for _ in range(self.steps)
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Attention without paged attention."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.inputs is None or self.k_cache is None or self.v_cache is None or self.model is None:
            raise RuntimeError("Baseline paged attention not initialized")

        seq_ptr = 0
        with nvtx_range("baseline_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                for query in self.inputs:
                    qkv = torch.matmul(query, self.model.in_proj_weight.T) + self.model.in_proj_bias
                    q, k, v = qkv.chunk(3, dim=-1)
                    k = k.view(self.batch_size, 1, self.num_heads, self.head_dim)
                    v = v.view(self.batch_size, 1, self.num_heads, self.head_dim)
                    self.k_cache[:, seq_ptr : seq_ptr + 1, :, :] = k
                    self.v_cache[:, seq_ptr : seq_ptr + 1, :, :] = v
                    seq_ptr += 1
                    k_all = self.k_cache[:, :seq_ptr, :, :].clone()
                    v_all = self.v_cache[:, :seq_ptr, :, :].clone()
                    q_heads = q.view(self.batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                    k_heads = k_all.permute(0, 2, 1, 3).contiguous()
                    v_heads = v_all.permute(0, 2, 1, 3).contiguous()
                    _ = torch.nn.functional.scaled_dot_product_attention(
                        q_heads, k_heads, v_heads, is_causal=False
                    )
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.k_cache = None
        self.v_cache = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=6,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.k_cache is None or self.v_cache is None:
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
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
