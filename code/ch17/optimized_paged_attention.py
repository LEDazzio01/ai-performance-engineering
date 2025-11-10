"""optimized_paged_attention.py - Optimized paged attention in inference/profiling.

Demonstrates paged attention for efficient KV cache management.
Paged attention: Uses non-contiguous pages for efficient memory management.
Reduces fragmentation and improves memory utilization for variable-length sequences.
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

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from common.python.paged_attention import PagedKVCache, PagedAttentionConfig

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")

class OptimizedPagedAttentionBenchmark(Benchmark):
    """Optimized: Paged attention for efficient KV cache management.
    
    Paged attention: Uses non-contiguous pages for efficient memory management.
    Reduces fragmentation and improves memory utilization for variable-length sequences.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.kv_cache = None
        self.inputs = None
        self.batch_size = 1
        self.sequence_length = 1
        self.hidden_dim = 256
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
    
    def setup(self) -> None:
        """Setup: Initialize model and paged KV cache."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: Paged attention - non-contiguous page-based storage
        
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        ).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()
        
        cache_config = PagedAttentionConfig(
            batch_size=self.batch_size,
            page_size=16,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            device=self.device,
            dtype=self.model.in_proj_weight.dtype,
        )
        self.kv_cache = PagedKVCache(cache_config)
        
        # Simulate autoregressive generation (single sequence for cache demo)
        input_dtype = next(self.model.parameters()).dtype
        self.inputs = [
            torch.randn(
                self.batch_size,
                self.sequence_length,
                self.hidden_dim,
                device=self.device,
                dtype=input_dtype,
            )
            for _ in range(64)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Paged attention."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Paged attention
                # Uses non-contiguous pages for efficient memory management
                
                model_dtype = next(self.model.parameters()).dtype
                
                for step, query in enumerate(self.inputs):
                    # Align precision with the compiled model
                    query = query.to(device=self.device, dtype=model_dtype)
                    
                    # Project inputs to Q/K/V so we can manage the cache manually
                    qkv = F.linear(query, self.model.in_proj_weight, self.model.in_proj_bias)
                    batch_size, seq_len = query.shape[:2]
                    qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
                    q, k, v = qkv.unbind(dim=2)  # Each: (batch, seq, num_heads, head_dim)
                    
                    q_heads = q.permute(0, 2, 1, 3).contiguous()
                    self.kv_cache.write(step, k.contiguous(), v.contiguous())
                    
                    k_all, v_all = self.kv_cache.get_kv(step + 1)
                    if k_all.numel() == 0:
                        continue
                    k_heads = k_all.permute(0, 2, 1, 3).contiguous()
                    v_heads = v_all.permute(0, 2, 1, 3).contiguous()
                    
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q_heads, k_heads, v_heads, is_causal=False
                    )
                    _ = attn_output.sum()

    
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
    return OptimizedPagedAttentionBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedPagedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: paged_attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
