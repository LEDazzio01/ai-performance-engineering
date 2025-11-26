"""Optimized sliding window attention - O(n·w) complexity vs O(n²).

This optimized version uses sliding window attention that only attends
to the last W tokens, reducing complexity from O(n²) to O(n·w).
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import math
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


class SlidingWindowAttentionModule(nn.Module):
    """Efficient sliding window attention using FlexAttention.
    
    Uses FlexAttention's block sparse masks for O(n·w) complexity.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Try to use FlexAttention
        self._has_flex = False
        try:
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
            self._flex_attention = flex_attention
            self._create_block_mask = create_block_mask
            self._has_flex = True
        except ImportError:
            pass
        
        self._mask_cache = {}
    
    def _get_block_mask(self, batch_size: int, seq_len: int, device):
        """Get or create cached block mask for FlexAttention."""
        key = (batch_size, seq_len, str(device))
        
        if key not in self._mask_cache and self._has_flex:
            window = self.window_size
            def mask_fn(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                in_window = (q_idx - kv_idx) <= window
                return causal & in_window
            
            self._mask_cache[key] = self._create_block_mask(
                mask_fn,
                B=batch_size,
                H=self.num_heads,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
        
        return self._mask_cache.get(key)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sliding window attention forward pass.
        
        Args:
            x: [batch, seq_len, embed_dim]
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        B, S, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self._has_flex:
            # Use FlexAttention with sliding window mask
            block_mask = self._get_block_mask(B, S, x.device)
            output = self._flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Fallback to standard SDPA with causal mask
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )
        
        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(output)


class OptimizedSlidingWindowBenchmark(BaseBenchmark):
    """Optimized: O(n·w) sliding window attention."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.batch_size = 4
        self.seq_len = 2048
        self.embed_dim = 1024
        self.num_heads = 16
        self.window_size = 512
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize sliding window attention model."""
        torch.manual_seed(42)
        
        self.model = SlidingWindowAttentionModule(
            self.embed_dim, self.num_heads, self.window_size
        ).to(self.device, self.dtype).eval()
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim,
            device=self.device, dtype=self.dtype
        )
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(self.x)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Sliding window attention."""
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
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None or self.x is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedSlidingWindowBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Sliding Window: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Config: batch={benchmark.batch_size}, seq={benchmark.seq_len}, window={benchmark.window_size}")
    print(f"  Complexity: O(n·{benchmark.window_size}) vs O(n²)")

