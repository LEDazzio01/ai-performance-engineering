"""optimized_attention_ilp.py - Optimized attention with high ILP.

Chapter 6: Occupancy and Instruction-Level Parallelism

Demonstrates ILP optimization via streaming chunks and concurrent execution.

FORWARD REFERENCE: This file uses F.scaled_dot_product_attention (SDPA),
which is covered in depth in Chapter 9 (arithmetic intensity, FlashAttention).
Here we use it to demonstrate ILP benefits from fused attention operations.
See ch9/baseline_sdpa_attention.py and ch9/optimized_sdpa_attention.py for
detailed SDPA analysis comparing unfused vs fused attention kernels.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import arch_config to apply Triton patch for sm_12x support
try:
    import arch_config  # noqa: F401
except ImportError:
    pass

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch6.workload_config import WORKLOAD


class OptimizedAttentionILPBenchmark(BaseBenchmark):
    """Optimized: attention with increased ILP via streaming chunks."""
    
    def __init__(self):
        super().__init__()
        self.qkv: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.input: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.batch = self.workload.attention_batch
        self.embed_dim = self.workload.attention_embed_dim
        self.num_heads = self.workload.attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.tokens = self.workload.attention_tokens
        self._last_sum: Optional[torch.Tensor] = None
        self.streams = [torch.cuda.Stream() for _ in range(2)]
        token_count = self.batch * self.tokens
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(token_count),
        )
    
    def setup(self) -> None:
        """Setup: Initialize optimized attention model."""
        torch.manual_seed(42)
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False).to(self.device).half()
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(self.device).half()
        self.input = torch.randn(
            self.batch,
            self.tokens,
            self.embed_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention with high ILP optimization."""
        assert self.qkv is not None and self.out_proj is not None and self.input is not None
        with self._nvtx_range("optimized_attention_ilp"):
            with torch.no_grad():
                chunks = self.input.chunk(len(self.streams), dim=0)
                self._last_sum = torch.zeros(1, device=self.device, dtype=self.input.dtype)

                for stream, chunk in zip(self.streams, chunks):
                    with torch.cuda.stream(stream):
                        qkv = self.qkv(chunk)
                        q, k, v = qkv.chunk(3, dim=-1)
                        q = q.reshape(chunk.size(0), chunk.size(1), self.num_heads, self.head_dim).transpose(1, 2)
                        k = k.reshape(chunk.size(0), chunk.size(1), self.num_heads, self.head_dim).transpose(1, 2)
                        v = v.reshape(chunk.size(0), chunk.size(1), self.num_heads, self.head_dim).transpose(1, 2)
                        attn = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            is_causal=False,
                        )
                        merged = attn.transpose(1, 2).reshape(chunk.size(0), chunk.size(1), self.embed_dim)
                        out = self.out_proj(merged)
                        self._last_sum += out.sum()

                self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.qkv = None
        self.out_proj = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self._last_sum is None:
            return "Attention output not computed"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedAttentionILPBenchmark()
