"""Baseline sliding window attention - Full O(n²) attention for comparison.

This baseline uses standard multi-head attention without any sparsity
to compare against the optimized sliding window implementation.
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


class FullAttentionModule(nn.Module):
    """Standard full attention (O(n²) complexity) - naive implementation.
    
    This uses explicit matmul/softmax/matmul to demonstrate the O(n²) cost
    that sliding window attention optimizes away.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full attention forward pass with explicit O(n²) computation.
        
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
        
        # Explicit O(n²) attention computation (not using Flash Attention)
        # scores: [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and weighted sum
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(output)


class BaselineSlidingWindowBenchmark(BaseBenchmark):
    """Baseline: Full O(n²) causal attention without sliding window."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.batch_size = 4
        self.seq_len = 2048
        self.embed_dim = 1024
        self.num_heads = 16
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize full attention model."""
        torch.manual_seed(42)
        
        self.model = FullAttentionModule(
            self.embed_dim, self.num_heads
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
        """Benchmark: Full attention."""
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
    return BaselineSlidingWindowBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Full Attention: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Config: batch={benchmark.batch_size}, seq={benchmark.seq_len}, dim={benchmark.embed_dim}")

