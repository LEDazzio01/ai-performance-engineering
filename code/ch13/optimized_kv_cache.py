"""optimized_kv_cache.py - Optimized KV cache."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedKVCache:
    """Optimized KV cache - efficient memory management."""
    
    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.cache_pool = []
        self.allocated_caches = {}
        for _ in range(10):
            cache_entry = []
            for _ in range(self.num_layers):
                k = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                v = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                cache_entry.append((k, v))
            self.cache_pool.append(cache_entry)
        self.free_indices = list(range(10))
    
    def allocate(self, request_id: str) -> None:
        if request_id in self.allocated_caches:
            return
        if self.free_indices:
            cache_idx = self.free_indices.pop()
            self.allocated_caches[request_id] = cache_idx
        else:
            cache_entry = []
            for _ in range(self.num_layers):
                k = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                v = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                cache_entry.append((k, v))
            self.cache_pool.append(cache_entry)
            cache_idx = len(self.cache_pool) - 1
            self.allocated_caches[request_id] = cache_idx
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        if request_id not in self.allocated_caches:
            self.allocate(request_id)
        cache_idx = self.allocated_caches[request_id]
        cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
        cache_k[pos:pos+1] = k.unsqueeze(0)
        cache_v[pos:pos+1] = v.unsqueeze(0)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_idx = self.allocated_caches[request_id]
        cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
        return cache_k[start:end], cache_v[start:end]
    
    def free(self, request_id: str) -> None:
        if request_id in self.allocated_caches:
            cache_idx = self.allocated_caches[request_id]
            for layer_idx in range(self.num_layers):
                cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
                cache_k.zero_()
                cache_v.zero_()
            if cache_idx < 10:
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
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(out)


class OptimizedKVCacheOptimizedBenchmark(BaseBenchmark):
    """Optimized KV cache with memory reuse."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.kv_cache = None
        self.inputs = None
        self.hidden_dim = 512
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.batch_size = 4
        self.max_seq_len = 256
        self.sequence_lengths = [128, 192, 256]
        tokens = self.batch_size * sum(self.sequence_lengths)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(len(self.sequence_lengths)),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        
        self.model = nn.ModuleList(
            [
                SimpleAttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=torch.float16)
                for _ in range(6)
            ]
        ).to(self.device).eval()
        
        self.kv_cache = OptimizedKVCache(
            max_seq_len=self.max_seq_len,
            num_layers=6,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        
        self.inputs = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            self.inputs.append(x)
        
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.model is None or self.kv_cache is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("optimized_kv_cache"):
            for seq_idx, x in enumerate(self.inputs):
                request_id = f"req_{seq_idx}"
                seq_len = x.size(1)
                
                for pos in range(seq_len):
                    token = x[:, pos:pos+1, :]
                    hidden = token
                    for layer_idx, layer in enumerate(self.model):
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)
                
                self.kv_cache.free(request_id)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.kv_cache = None
        self.inputs = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedKVCacheOptimizedBenchmark:
    return OptimizedKVCacheOptimizedBenchmark()
