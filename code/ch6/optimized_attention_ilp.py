"""optimized_attention_ilp.py - Optimized attention with high ILP.

Demonstrates attention operations optimized for instruction-level parallelism.
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

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch6.workload_config import WORKLOAD, is_smoke_test

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedAttentionILPBenchmark(Benchmark):
    """Optimized: Attention with high ILP optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.qkv = None
        self.out_proj = None
        self.input = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.batch = self.workload.attention_batch
        self.embed_dim = self.workload.attention_embed_dim
        self.num_heads = self.workload.attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.tokens = self.workload.attention_tokens_for_mode(self.smoke_test)
        self._last_sum = None
        self.streams = [torch.cuda.Stream() for _ in range(2)]
    
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
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention with high ILP optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_attention_ilp", enable=enable_nvtx):
            with torch.no_grad():
                assert self.qkv is not None and self.out_proj is not None
                chunks = self.input.chunk(len(self.streams), dim=0)
                self._last_sum = torch.zeros(1, device=self.device, dtype=self.input.dtype)

                for stream, chunk in zip(self.streams, chunks):
                    with torch.cuda.stream(stream):
                        qkv = self.qkv(chunk)
                        q, k, v = qkv.chunk(3, dim=-1)
                        # reshape() tolerates non-contiguous chunks from the stream split
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

                torch.cuda.synchronize()

    
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
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self._last_sum is None:
            return "Attention output not computed"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAttentionILPBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    
    print(f"\nOptimized Attention ILP: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Optimized attention operations maximize instruction-level parallelism")
