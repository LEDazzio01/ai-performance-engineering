"""baseline kv cache - Baseline implementation. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineKVCacheAttention(nn.Module):
    """Baseline attention without efficient KV cache management."""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Single projection for Q, K, V
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass without efficient KV cache management."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Baseline: No efficient KV cache management
        # If cache exists, inefficiently concatenate (reallocates memory)
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            # Inefficient: creates new tensors each time
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        
        # Return new cache (inefficient: copies entire cache)
        new_kv_cache = (k, v)
        return output, new_kv_cache


class KvCacheBenchmark(Benchmark):
    """Baseline implementation with inefficient KV cache management."""

    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.kv_cache = None

    def setup(self) -> None:
        """Setup: Initialize model and data."""
        self.model = BaselineKVCacheAttention(hidden_dim=256, num_heads=8).to(self.device).eval()
        # Create sequence of inputs (simulating autoregressive generation)
        self.inputs = [torch.randn(1, 1, 256, device=self.device) for _ in range(300)]  # Match optimized scale
        self.kv_cache = None
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Run computation with inefficient KV cache."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_kv_cache", enable=enable_nvtx):
            self.kv_cache = None
            for x in self.inputs:
                output, self.kv_cache = self.model(x, self.kv_cache)
                # Sync to ensure accurate timing
                torch.cuda.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        del self.model, self.inputs, self.kv_cache
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return KvCacheBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nResult: {result.mean_ms:.3f} ms")
