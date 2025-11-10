"""optimized_paged_attention.py - Optimized paged attention.

Demonstrates paged attention for efficient KV cache management.
Paged attention: Uses non-contiguous pages for efficient memory management.
Reduces fragmentation and improves memory utilization for variable-length sequences.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.python.compile_utils import enable_tf32

from typing import Optional, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

class PagedKVCache:
    """Paged KV cache using reusable page buffers."""

    def __init__(
        self,
        batch_size: int,
        page_size: int,
        num_heads: int,
        head_dim: int,
        total_positions: int,
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.total_pages = (total_positions + page_size - 1) // page_size
        shape = (batch_size, self.total_pages, page_size, num_heads, head_dim)
        self.k_pages = torch.zeros(shape, device=device, dtype=torch.float32)
        self.v_pages = torch.zeros_like(self.k_pages)
        self.length = 0

    def write(self, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        page_idx = position // self.page_size
        offset = position % self.page_size
        self.k_pages[:, page_idx, offset, :, :] = k[:, 0, :, :].to(self.device, dtype=torch.float32)
        self.v_pages[:, page_idx, offset, :, :] = v[:, 0, :, :].to(self.device, dtype=torch.float32)
        self.length = max(self.length, position + 1)

    def reset(self) -> None:
        self.length = 0

    def get_kv(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        length = min(length, self.length)
        if length == 0:
            empty = torch.empty(self.batch_size, 0, self.num_heads, self.head_dim, device=self.device)
            return empty, empty
        full_pages = length // self.page_size
        tail = length % self.page_size

        def _assemble(pages: torch.Tensor) -> torch.Tensor:
            chunks = []
            if full_pages > 0:
                chunks.append(
                    pages[:, :full_pages]
                    .reshape(self.batch_size, full_pages * self.page_size, self.num_heads, self.head_dim)
                )
            if tail > 0:
                chunks.append(pages[:, full_pages, :tail, :, :])
            return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)

        return _assemble(self.k_pages), _assemble(self.v_pages)

class OptimizedPagedAttentionBenchmark(Benchmark):
    """Optimized: Paged attention for efficient KV cache management.
    
    Paged attention: Uses non-contiguous pages for efficient memory management.
    Reduces fragmentation and improves memory utilization for variable-length sequences.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.kv_cache = None
        self.inputs = None
        self.hidden_dim = 512
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.batch_size = 4
        self.page_size = 32
        self.steps = 512
    
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
        # Paged attention uses pages for efficient memory management
        
        # Simple attention model
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        ).to(self.device).eval()
        
        # Optimization: Paged KV cache (paged attention)
        # Uses non-contiguous pages for efficient memory management
        self.kv_cache = PagedKVCache(
            batch_size=self.batch_size,
            page_size=self.page_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            total_positions=self.steps,
            device=self.device,
        )
        
        # Simulate autoregressive generation
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
            for _ in range(self.steps)
        ]
        torch.cuda.synchronize()
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, hidden) into (B, heads, T, head_dim)."""
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
    
    def _project_qkv(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs into Q, K, V using the MHA weights."""
        qkv = F.linear(query, self.model.in_proj_weight, self.model.in_proj_bias)
        return qkv.chunk(3, dim=-1)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Paged attention."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                if self.kv_cache:
                    self.kv_cache.reset()
                # Optimization: Paged attention
                # Uses non-contiguous pages for efficient memory management
                # Reduces fragmentation and improves memory utilization
                
                for step, query in enumerate(self.inputs):
                    q_proj, k_proj, v_proj = self._project_qkv(query)
                    q_heads = self._split_heads(q_proj)
                    k_new = self._split_heads(k_proj).permute(0, 2, 1, 3)
                    v_new = self._split_heads(v_proj).permute(0, 2, 1, 3)
                    
                    # Store in paged cache (paged attention)
                    assert self.kv_cache is not None
                    self.kv_cache.write(step, k_new, v_new)
                    
                    # Retrieve K, V from reusable pages
                    k_all, v_all = self.kv_cache.get_kv(step + 1)
                    if k_all.shape[1] == 0:
                        continue
                    
                    k_heads = k_all.permute(0, 2, 1, 3).contiguous()
                    v_heads = v_all.permute(0, 2, 1, 3).contiguous()
                    
                    attn_scores = torch.matmul(
                        q_heads, k_heads.transpose(-2, -1)
                    ) / math.sqrt(self.head_dim)
                    attn_probs = torch.softmax(attn_scores, dim=-1)
                    context = torch.matmul(attn_probs, v_heads)
                    
                    _ = context.permute(0, 2, 1, 3).reshape(query.size(0), 1, self.hidden_dim)

    
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
