"""optimized_kv_cache_paged.py - Paged KV cache implementation (optimized).

Paged KV cache allocates memory in fixed-size pages, enabling efficient memory reuse
and eliminating fragmentation. Pages are allocated on-demand and reused across sequences.

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
    BenchmarkHarness,
    BenchmarkMode,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")

class PagedKVCache:
    """Paged KV cache - allocates memory in fixed-size pages for efficient reuse."""
    
    def __init__(self, page_size: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device, max_pages: int = 1000):
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.max_pages = max_pages
        
        # Pre-allocate pool of pages (reusable)
        self.page_pool: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.allocated_pages: dict[str, list[tuple[int, int]]] = {}  # request_id -> [(page_idx, offset_in_page)]
        
        # Allocate pages upfront for reuse
        for _ in range(max_pages):
            k_page = torch.zeros(page_size, num_heads, head_dim, dtype=dtype, device=device)
            v_page = torch.zeros(page_size, num_heads, head_dim, dtype=dtype, device=device)
            self.page_pool.append((k_page, v_page))
        
        self.free_pages = list(range(max_pages))
    
    def allocate_page(self) -> int:
        """Allocate a page from the pool."""
        if not self.free_pages:
            raise RuntimeError("Out of pages")
        return self.free_pages.pop()
    
    def free_page(self, page_idx: int) -> None:
        """Return a page to the pool."""
        self.free_pages.append(page_idx)
    
    def allocate(self, request_id: str, seq_len: int) -> None:
        """Allocate pages for a sequence (only what's needed)."""
        if request_id in self.allocated_pages:
            return
        
        num_pages_needed = (seq_len + self.page_size - 1) // self.page_size
        self.allocated_pages[request_id] = []
        
        for _ in range(num_pages_needed):
            page_idx = self.allocate_page()
            self.allocated_pages[request_id].append((page_idx, 0))
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        """Append keys/values to cache using paged allocation.
        
        Args:
            k, v: Shape [num_heads, head_dim] (single token)
        """
        if request_id not in self.allocated_pages:
            # Estimate sequence length, allocate pages
            estimated_len = pos + 100  # Small buffer
            self.allocate(request_id, estimated_len)
        
        # Find which page and offset to use
        page_idx, offset = self.allocated_pages[request_id][pos // self.page_size]
        local_offset = pos % self.page_size
        
        cache_k, cache_v = self.page_pool[page_idx]
        # k/v shape: [num_heads, head_dim]
        # Store as [1, num_heads, head_dim] in cache at local_offset
        cache_k[local_offset:local_offset+1] = k.unsqueeze(0)  # Add sequence dimension
        cache_v[local_offset:local_offset+1] = v.unsqueeze(0)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys/values from pages."""
        # Collect from multiple pages if needed
        k_list, v_list = [], []
        
        for pos in range(start, end):
            page_idx, _ = self.allocated_pages[request_id][pos // self.page_size]
            local_offset = pos % self.page_size
            cache_k, cache_v = self.page_pool[page_idx]
            k_list.append(cache_k[local_offset:local_offset + 1])
            v_list.append(cache_v[local_offset:local_offset + 1])
        
        if k_list:
            k = torch.cat(k_list, dim=0)
            v = torch.cat(v_list, dim=0)
            return k, v
        else:
            # Return empty tensors
            k = torch.zeros(0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            v = torch.zeros(0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            return k, v
    
    def free(self, request_id: str) -> None:
        """Free pages for a sequence (return to pool)."""
        if request_id in self.allocated_pages:
            for page_idx, _ in self.allocated_pages[request_id]:
                self.free_page(page_idx)
            del self.allocated_pages[request_id]

class SimpleAttentionLayer(nn.Module):
    """Simple attention layer for KV cache demo."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
    
    def forward(self, x: torch.Tensor, kv_cache: PagedKVCache, request_id: str, layer_idx: int, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        # q, k, v: [batch_size, seq_len, hidden_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update KV cache (paged: allocate only what's needed)
        # k/v shape: [batch_size, num_heads, seq_len, head_dim]
        # Extract per-head tokens: [num_heads, head_dim] for position cache_pos
        k_to_cache = k[:, :, 0, :].transpose(0, 1)  # [num_heads, batch_size, head_dim]
        v_to_cache = v[:, :, 0, :].transpose(0, 1)  # [num_heads, batch_size, head_dim]
        k_single = k_to_cache[:, 0, :]  # [num_heads, head_dim] - use first batch item
        v_single = v_to_cache[:, 0, :]  # [num_heads, head_dim]
        kv_cache.append(request_id, layer_idx, k_single, v_single, cache_pos)
        
        # Get cached keys/values for attention
        if cache_pos > 0:
            cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos)
            # cached_k/v shape: [cache_pos, num_heads, head_dim]
            # Reshape to [batch_size, num_heads, cache_pos, head_dim]
            cached_k = cached_k.permute(1, 0, 2)  # [num_heads, cache_pos, head_dim]
            cached_v = cached_v.permute(1, 0, 2)  # [num_heads, cache_pos, head_dim]
            cached_k = cached_k.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, num_heads, cache_pos, head_dim]
            cached_v = cached_v.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, num_heads, cache_pos, head_dim]
            k = torch.cat([cached_k, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([cached_v, v], dim=2)
        
        # Simple attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(out)

class OptimizedKVCachePagedBenchmark(Benchmark):
    """Paged KV cache optimization - efficient memory reuse."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.kv_cache = None
        self.inputs = None
        self.page_size = 32  # Smaller page size
        self.num_layers = 2  # Reduced layers
        self.num_heads = 4  # Reduced heads
        self.head_dim = 64
        self.hidden_dim = self.num_heads * self.head_dim
        self.batch_size = 1  # Single batch for simplicity
        self.sequence_lengths = [64, 128]  # Shorter sequences for faster benchmarking
    
    def setup(self) -> None:
        """Setup: Initialize model and paged KV cache."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        # Create simple model with attention layers
        layers = []
        for _ in range(self.num_layers):
            layers.append(SimpleAttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=torch.float16))
        self.model = nn.Sequential(*layers).to(self.device).eval()
        
        # Paged KV cache (allocates pages on-demand, reuses them)
        self.kv_cache = PagedKVCache(
            page_size=self.page_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=torch.float16,
            device=self.device,
            max_pages=1000
        )
        
        # Prepare inputs (different sequence lengths - paged cache handles efficiently)
        self.inputs = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            self.inputs.append(x)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - paged KV cache with efficient allocation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_kv_cache_paged", enable=enable_nvtx):
            # Process multiple sequences with different lengths
            for seq_idx, x in enumerate(self.inputs):
                request_id = f"req_{seq_idx}"
                seq_len = x.size(1)
                
                # Allocate pages only for this sequence length (efficient)
                self.kv_cache.allocate(request_id, seq_len)
                
                # Process sequence token by token (decoding scenario)
                for pos in range(seq_len):
                    # Process single token
                    token = x[:, pos:pos+1, :]
                    
                    # Forward through all layers with KV cache
                    hidden = token
                    for layer_idx, layer in enumerate(self.model):
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)
                
                # Free cache after sequence (pages returned to pool for reuse)
                self.kv_cache.free(request_id)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.kv_cache, self.inputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,  # Reduced iterations
            warmup=3,  # Reduced warmup
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedKVCachePagedBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Paged KV Cache: {result.mean_ms:.3f} ms")

