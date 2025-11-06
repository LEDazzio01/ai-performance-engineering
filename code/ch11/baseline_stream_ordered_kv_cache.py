"""baseline_stream_ordered_kv_cache.py - Baseline stream-ordered allocator without KV cache optimization.

Demonstrates stream-ordered memory allocation without KV cache reuse.
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

from typing import Optional, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class BaselineStreamOrderedKVCacheBenchmark(Benchmark):
    """Baseline: stream-ordered allocator without KV cache reuse."""

    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.stream = None
        self.kv_cache = None
        self.batch_size = None

    def setup(self) -> None:
        """Setup: Initialize model and stream-ordered memory."""
        # Simple attention layer
        self.model = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.model = self.model.to(self.device).eval()

        # Create stream for stream-ordered allocation
        self.stream = torch.cuda.Stream()

        # Baseline: allocate KV cache per step (no reuse)
        self.batch_size = 2
        seq_len = 64
        hidden_dim = 256

        # Simulate autoregressive generation steps
        self.inputs = [
            torch.randn(self.batch_size, 1, hidden_dim, device=self.device)
            for _ in range(seq_len)
        ]

        # Baseline: No pre-allocated cache (allocate per step)
        self.kv_cache = None

    def benchmark_fn(self) -> None:
        """Benchmark: stream-ordered allocation without KV cache reuse."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_stream_ordered_kv_cache", enable=enable_nvtx):
            with torch.cuda.stream(self.stream):
                # Baseline: allocate new KV cache at each step (no reuse)
                step = 0
                for step_input in self.inputs:
                    # Allocate new memory for KV cache each step
                    # This is inefficient - no reuse of previous cache
                    # Each step allocates a larger cache and copies old data
                    current_seq_len = step + 1
                    k_cache = torch.empty(
                        self.batch_size, 8, current_seq_len, 32,
                        device=self.device, dtype=torch.float16
                    )
                    v_cache = torch.empty(
                        self.batch_size, 8, current_seq_len, 32,
                        device=self.device, dtype=torch.float16
                    )

                    # Compute attention (simplified)
                    with torch.no_grad():
                        # Use MultiheadAttention's forward to get Q, K, V properly
                        # step_input shape: (batch_size, 1, hidden_dim)
                        # We'll use the model's internal projection
                        attn_output, _ = self.model(step_input, step_input, step_input, need_weights=False)
                        
                        # For KV cache, we need to manually extract K, V from the projection
                        # Simplified: project input to get K, V for caching
                        # Use the model's in_proj_weight to project
                        qkv = torch.nn.functional.linear(step_input, self.model.in_proj_weight, self.model.in_proj_bias)
                        qkv = qkv.view(self.batch_size, 1, 3, -1)  # (batch, 1, 3, embed_dim)
                        q, k, v = qkv.chunk(3, dim=2)  # Each: (batch, 1, 1, embed_dim)
                        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)  # (batch, 1, embed_dim)
                        
                        # Reshape for multi-head: (batch, 1, embed_dim) -> (batch, num_heads, 1, head_dim)
                        num_heads = self.model.num_heads
                        head_dim = self.model.embed_dim // num_heads
                        k = k.reshape(self.batch_size, 1, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)
                        v = v.reshape(self.batch_size, 1, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, 1, head_dim)

                        # Allocate new cache and copy old data (inefficient - no reuse)
                        if step > 0 and self.kv_cache is not None:
                            # Copy old cache to new (memory overhead from reallocation)
                            old_k, old_v = self.kv_cache
                            old_seq_len = old_k.shape[2]
                            k_cache[:, :, :old_seq_len, :] = old_k
                            v_cache[:, :, :old_seq_len, :] = old_v

                        # Append new K/V to new cache
                        k_cache[:, :, step:step+1, :] = k
                        v_cache[:, :, step:step+1, :] = v

                        self.kv_cache = (k_cache, v_cache)
                        step += 1

                self.stream.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
        self.kv_cache = None
        if self.stream:
            self.stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=3,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineStreamOrderedKVCacheBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Stream-Ordered KV Cache: {result.mean_ms:.3f} ms")
    print(" Note: Allocates new KV cache each step (no reuse)")
