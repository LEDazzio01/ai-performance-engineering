"""Baseline sliding window attention - Naive O(n²) explicit attention.

This baseline uses explicit matmul/softmax/matmul operations which have:
- O(n²) memory for attention scores matrix
- Multiple kernel launches (matmul, softmax, matmul)
- No memory optimization

Compare with optimized_sliding_window_bench.py which uses Flash Attention
via scaled_dot_product_attention for O(n) memory and fused kernels.
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
from typing import Optional

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class NaiveAttentionModule(nn.Module):
    """Naive O(n²) attention using explicit matmul operations.
    
    This is the slow baseline that Flash Attention optimizes.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Naive O(n²) attention with explicit matmul operations."""
        B, S, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Explicit O(n²) attention - creates full S×S attention matrix
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and weighted sum
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(output)


class BaselineSlidingWindowBenchmark(BaseBenchmark):
    """Baseline: Naive O(n²) attention (slow)."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.batch_size = 4
        self.seq_len = 4096  # Longer sequence to show difference
        self.embed_dim = 1024
        self.num_heads = 16
        self.dtype = torch.float16  # Use float16 for better Flash Attention support
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize naive attention model."""
        torch.manual_seed(42)
        
        self.model = NaiveAttentionModule(
            self.embed_dim, self.num_heads
        ).to(self.device, self.dtype).eval()
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim,
            device=self.device, dtype=self.dtype
        )
        
        # Proper warmup to avoid cold cache effects
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(self.x)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Naive attention."""
        with torch.no_grad():
            output = self.model(self.x)
            self._last = float(output.sum())
            self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.x = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.x is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineSlidingWindowBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Naive Attention: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Config: batch={benchmark.batch_size}, seq={benchmark.seq_len}, dim={benchmark.embed_dim}")
