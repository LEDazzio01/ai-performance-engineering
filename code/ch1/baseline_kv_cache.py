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


HIDDEN_DIM = 512
NUM_HEADS = 8
TOKENS_PER_STEP = 2
DECODE_STEPS = 512  # results in 1024 cached tokens


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
        self.hidden_dim = HIDDEN_DIM
        self.num_heads = NUM_HEADS
        self.tokens_per_step = TOKENS_PER_STEP
        self.decode_steps = DECODE_STEPS
        self.batch_size = 1

    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        self.model = BaselineKVCacheAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
        ).to(self.device).eval()

        # Pre-generate all tokens so each benchmark iteration replays identical work
        samples = torch.randn(
            self.decode_steps,
            self.batch_size,
            self.tokens_per_step,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.inputs = [samples[idx].contiguous() for idx in range(self.decode_steps)]
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
                # Sync to ensure accurate timing of each autoregressive step
                torch.cuda.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        del self.model, self.inputs, self.kv_cache
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=15,
            warmup=3,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.kv_cache is None:
            return "KV cache not produced"
        k_cache, v_cache = self.kv_cache
        expected_tokens = self.decode_steps * self.tokens_per_step
        if k_cache.shape[2] != expected_tokens or v_cache.shape[2] != expected_tokens:
            return (
                f"Unexpected cache shape: "
                f"K={k_cache.shape}, V={v_cache.shape}, expected tokens={expected_tokens}"
            )
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
    print(f"\nResult: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
