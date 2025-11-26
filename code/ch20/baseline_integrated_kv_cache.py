"""baseline_integrated_kv_cache.py - Baseline integrated KV cache (baseline).

Baseline KV cache integration in inference pipeline without optimizations.
Uses naive cache allocation and access patterns.

Implements BaseBenchmark for harness integration.
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
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class NaiveKVCache:
    """Naive KV cache - simple allocation."""
    
    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    
    def allocate(self, request_id: str) -> None:
        if request_id not in self.cache:
            self.cache[request_id] = []
            for _ in range(self.num_layers):
                k = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                v = torch.zeros(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                self.cache[request_id].append((k, v))
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        if request_id not in self.cache:
            self.allocate(request_id)
        cache_k, cache_v = self.cache[request_id][layer_idx]
        cache_k[pos:pos+1] = k.unsqueeze(0)
        cache_v[pos:pos+1] = v.unsqueeze(0)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_k, cache_v = self.cache[request_id][layer_idx]
        return cache_k[start:end], cache_v[start:end]
    
    def free(self, request_id: str) -> None:
        if request_id in self.cache:
            del self.cache[request_id]


class AttentionLayer(nn.Module):
    """Attention layer with KV cache."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
    
    def forward(self, x: torch.Tensor, kv_cache: NaiveKVCache, request_id: str, layer_idx: int, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_to_cache = k[:, :, 0, :].transpose(0, 1)
        v_to_cache = v[:, :, 0, :].transpose(0, 1)
        k_single = k_to_cache[:, 0, :]
        v_single = v_to_cache[:, 0, :]
        kv_cache.append(request_id, layer_idx, k_single, v_single, cache_pos)
        
        if cache_pos > 0:
            cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos)
            # Touch cached tensors to simulate reads without rebuilding large attention matrices
            _ = cached_k.sum()
            _ = cached_v.sum()
        
        return self.proj(x)


class BaselineIntegratedKVCacheBenchmark(BaseBenchmark):
    """Baseline integrated KV cache - naive implementation."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model = None
        self.kv_cache = None
        self.inputs = None
        self.max_seq_len = 8192
        self.num_layers = 1
        self.num_heads = 2
        self.head_dim = 32
        self.hidden_dim = self.num_heads * self.head_dim
        self.batch_size = 1
        self.sequence_lengths = [512, 1024, 2048]
    
    def setup(self) -> None:
        """Setup: Initialize baseline model with naive KV cache."""
        torch.manual_seed(42)
        
        layers = []
        for _ in range(self.num_layers):
            layers.append(AttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=torch.float32))
        self.model = nn.Sequential(*layers).to(self.device).eval()
        
        self.kv_cache = NaiveKVCache(
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=torch.float32,
            device=self.device
        )
        
        self.inputs = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=torch.float32)
            self.inputs.append(x)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - baseline integrated KV cache."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_integrated_kv_cache", enable=enable_nvtx):
            for seq_idx, x in enumerate(self.inputs):
                request_id = f"req_{seq_idx}"
                seq_len = x.size(1)
                self.kv_cache.allocate(request_id)
                
                for pos in range(seq_len):
                    token = x[:, pos:pos+1, :]
                    hidden = token
                    for layer_idx, layer in enumerate(self.model):
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)
                
                self.kv_cache.free(request_id)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.kv_cache, self.inputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=3,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineIntegratedKVCacheBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Integrated KV Cache: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
