"""optimized_kv_cache.py - Optimized KV cache.

Optimized KV cache with efficient memory management and reuse.
Better memory efficiency and faster access patterns.

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
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

class OptimizedKVCache:
    """Optimized KV cache - efficient memory management."""
    
    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Optimized: Pre-allocate reusable cache pools
        self.cache_pool = []
        self.allocated_caches = {}  # request_id -> cache_index
        
        # Pre-allocate cache pool for reuse
        for _ in range(10):  # Pool of 10 reusable caches
            cache_entry = []
            for _ in range(self.num_layers):
                k = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                v = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                cache_entry.append((k, v))
            self.cache_pool.append(cache_entry)
        
        self.free_indices = list(range(10))
    
    def allocate(self, request_id: str) -> None:
        """Allocate cache from pool (optimized reuse)."""
        if request_id in self.allocated_caches:
            return
        
        if self.free_indices:
            cache_idx = self.free_indices.pop()
            self.allocated_caches[request_id] = cache_idx
        else:
            # Allocate new cache if pool exhausted
            cache_entry = []
            for _ in range(self.num_layers):
                k = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                v = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                cache_entry.append((k, v))
            self.cache_pool.append(cache_entry)
            cache_idx = len(self.cache_pool) - 1
            self.allocated_caches[request_id] = cache_idx
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        """Append keys/values to cache."""
        if request_id not in self.allocated_caches:
            self.allocate(request_id)
        cache_idx = self.allocated_caches[request_id]
        cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
        cache_k[pos:pos+1] = k.unsqueeze(0)
        cache_v[pos:pos+1] = v.unsqueeze(0)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys/values."""
        cache_idx = self.allocated_caches[request_id]
        cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
        return cache_k[start:end], cache_v[start:end]
    
    def free(self, request_id: str) -> None:
        """Free cache and return to pool (optimized reuse)."""
        if request_id in self.allocated_caches:
            cache_idx = self.allocated_caches[request_id]
            # Clear cache by zeroing (for reuse)
            for layer_idx in range(self.num_layers):
                cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
                cache_k.zero_()
                cache_v.zero_()
            # Return to pool
            if cache_idx < 10:  # Only return if from original pool
                self.free_indices.append(cache_idx)
            del self.allocated_caches[request_id]

class SimpleAttentionLayer(nn.Module):
    """Simple attention layer for KV cache demo."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
    
    def forward(self, x: torch.Tensor, kv_cache: OptimizedKVCache, request_id: str, layer_idx: int, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if cache_pos == 0:
            kv_cache.allocate(request_id)
        
        k_to_cache = k[:, :, 0, :].transpose(0, 1)
        v_to_cache = v[:, :, 0, :].transpose(0, 1)
        k_single = k_to_cache[:, 0, :]
        v_single = v_to_cache[:, 0, :]
        kv_cache.append(request_id, layer_idx, k_single, v_single, cache_pos)
        
        if cache_pos > 0:
            cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos)
            cached_k = cached_k.permute(1, 0, 2)
            cached_v = cached_v.permute(1, 0, 2)
            cached_k = cached_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
            cached_v = cached_v.unsqueeze(0).expand(batch_size, -1, -1, -1)
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(out)

class OptimizedKVCacheOptimizedBenchmark(Benchmark):
    """Optimized KV cache - efficient memory reuse."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.kv_cache = None
        self.inputs = None
        self.max_seq_len = 256
        self.num_layers = 2
        self.num_heads = 4
        self.head_dim = 64
        self.hidden_dim = self.num_heads * self.head_dim
        self.batch_size = 1
        self.sequence_lengths = [32, 64]
    
    def setup(self) -> None:
        """Setup: Initialize model and optimized KV cache."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        layers = []
        for _ in range(self.num_layers):
            layers.append(SimpleAttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=torch.float16))
        self.model = nn.Sequential(*layers).to(self.device).eval()
        
        self.kv_cache = OptimizedKVCache(
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=torch.float16,
            device=self.device
        )
        
        self.inputs = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            self.inputs.append(x)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - optimized KV cache."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_kv_cache", enable=enable_nvtx):
            for seq_idx, x in enumerate(self.inputs):
                request_id = f"req_{seq_idx}"
                seq_len = x.size(1)
                
                for pos in range(seq_len):
                    token = x[:, pos:pos+1, :]
                    hidden = token
                    for layer_idx, layer in enumerate(self.model):
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)
                
                # Optimized: cache returned to pool for reuse
                self.kv_cache.free(request_id)

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.kv_cache, self.inputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
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
    return OptimizedKVCacheOptimizedBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized KV Cache: {result.mean_ms:.3f} ms")

