"""baseline_attention.py - Naive attention implementation (baseline).

Chapter 10: Blackwell Software Optimizations

This baseline demonstrates a naive attention implementation with:
- Explicit Q, K, V matrix multiplications
- Softmax with explicit exp/sum operations
- Multiple separate kernel launches
- No memory-efficient attention algorithms

The naive approach has O(n²) memory complexity and poor cache efficiency.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineAttentionBenchmark(BaseBenchmark):
    """Naive attention with explicit matrix operations."""

    def __init__(self):
        super().__init__()
        self.query: Optional[torch.Tensor] = None
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        # Larger sizes to show optimization benefits
        self.batch_size = 16
        self.seq_len = 512
        self.hidden_dim = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def setup(self) -> None:
        torch.manual_seed(42)
        
        # Create Q, K, V tensors in FP32 (no tensor cores)
        self.query = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float32
        )
        self.key = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float32
        )
        self.value = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float32
        )
        
        self._synchronize()
        tokens = float(self.batch_size * self.seq_len)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.batch_size),
        )

    def benchmark_fn(self) -> None:
        """Benchmark: Naive attention with explicit operations.
        
        Naive approach:
        1. Compute attention scores: QK^T (O(n²) memory)
        2. Scale scores
        3. Apply softmax (materializes full attention matrix)
        4. Compute output: Attention @ V
        
        This creates a full n×n attention matrix, causing:
        - O(n²) memory usage
        - Poor cache locality
        - Multiple kernel launches
        """
        with self._nvtx_range("baseline_attention_naive"):
            with torch.no_grad():
                # Naive: Explicit matrix multiplications
                # Q @ K^T -> (batch, heads, seq, seq) attention matrix
                attn_scores = torch.matmul(self.query, self.key.transpose(-2, -1))
                attn_scores = attn_scores * self.scale
                
                # Softmax over last dimension (materializes full attention matrix)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                
                # Attention @ V -> output
                _output = torch.matmul(attn_weights, self.value)
        self._synchronize()

    def teardown(self) -> None:
        self.query = None
        self.key = None
        self.value = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        if self.query is None or self.key is None or self.value is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaselineAttentionBenchmark:
    return BaselineAttentionBenchmark()
