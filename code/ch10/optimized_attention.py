"""optimized_attention.py - Flash Attention / SDPA optimized (optimized).

Chapter 10: Blackwell Software Optimizations

This optimized version uses PyTorch's scaled_dot_product_attention which:
- Automatically selects Flash Attention or Memory-Efficient Attention
- Uses FP16 for tensor core acceleration
- Fuses QK^T softmax and attention@V into a single kernel
- Has O(n) memory complexity (no materialized attention matrix)
- Optimized for GPU memory hierarchy

Key optimization: Memory-efficient attention algorithms avoid materializing
the full n×n attention matrix, dramatically reducing memory bandwidth.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


class OptimizedAttentionBenchmark(BaseBenchmark):
    """Optimized attention using Flash Attention/SDPA and FP16."""

    def __init__(self):
        super().__init__()
        self.query: Optional[torch.Tensor] = None
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        # Larger sizes to show tensor core optimization benefits
        self.batch_size = 16
        self.seq_len = 512
        self.hidden_dim = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)

        # Use FP16 for tensor core optimization
        # Create Q, K, V tensors in the format expected by SDPA
        self.query = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float16
        )
        self.key = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float16
        )
        self.value = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float16
        )
        
        self._synchronize()
        tokens = float(self.batch_size * self.seq_len)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.batch_size),
        )

    def benchmark_fn(self) -> None:
        """Benchmark: Optimized attention using SDPA/Flash Attention.
        
        Optimizations:
        1. FP16 precision for tensor core acceleration
        2. Flash Attention fuses all operations into single kernel
        3. No materialized attention matrix (O(n) memory)
        4. Tiled computation for cache efficiency
        5. Optimized for modern GPU memory hierarchy
        """
        with self._nvtx_range("optimized_attention_flash"):
            with torch.no_grad():
                # scaled_dot_product_attention automatically selects:
                # - Flash Attention V2 when available
                # - Memory-efficient attention as fallback
                _output = torch.nn.functional.scaled_dot_product_attention(
                    self.query, self.key, self.value,
                    dropout_p=0.0,
                    is_causal=False,
                )
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


def get_benchmark() -> OptimizedAttentionBenchmark:
    return OptimizedAttentionBenchmark()
