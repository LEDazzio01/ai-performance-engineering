"""optimized_kv_cache_management.py - Optimized KV cache management in multi-GPU.

Demonstrates efficient KV cache management across distributed GPUs.
KV cache management: Implements efficient pre-allocation, reuse, and synchronization.
Reduces memory overhead and improves performance.
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
import torch.distributed as dist

from common.python.compile_utils import enable_tf32
from common.python.compile_utils import compile_model
from ch4.gpu_requirements import skip_if_insufficient_gpus

from typing import Optional, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda")

class OptimizedKVCacheAttention:
    """Optimized attention with efficient KV cache management."""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Single projection for Q, K, V
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_pos: int = 0
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with efficient KV cache management."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Optimization: Efficient KV cache management
        # Write to pre-allocated cache at specific position (in-place)
        if k_cache is not None and v_cache is not None:
            pass
    # Ensure k and v match cache dtype before assignment
            k = k.to(k_cache.dtype)
            v = v.to(v_cache.dtype)
            k_cache[:, :, cache_pos:cache_pos+seq_len, :] = k
            v_cache[:, :, cache_pos:cache_pos+seq_len, :] = v
    # Use full cache for attention
            k_attn = k_cache[:, :, :cache_pos+seq_len, :]
            v_attn = v_cache[:, :, :cache_pos+seq_len, :]
        else:
            k_attn = k
            v_attn = v
        
        # Attention computation
        # Ensure q matches k_attn dtype to avoid dtype mismatch errors
        q = q.to(k_attn.dtype)
        scores = torch.matmul(q, k_attn.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        # Ensure attn matches v_attn dtype
        attn = attn.to(v_attn.dtype)
        output = torch.matmul(attn, v_attn)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        
        return output, k_cache, v_cache
class OptimizedKVCacheManagementBenchmark:
    """Optimized: Efficient KV cache management across GPUs.
    
        KV cache management: Implements efficient pre-allocation, in-place updates,
        and proper synchronization across distributed GPUs.
        """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None

        self.inputs = None
        self.k_cache = None
        self.v_cache = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self) -> None:
        """Setup: Initialize model and pre-allocated KV cache."""
        skip_if_insufficient_gpus()
        
        # Enable TF32 for faster matmul on Ampere+ GPUs
        enable_tf32()
        # Initialize distributed if available
        if dist.is_available() and torch.cuda.device_count() > 1:
            try:
                if not dist.is_initialized():
                    import os
                    if 'MASTER_ADDR' not in os.environ:
                        os.environ['MASTER_ADDR'] = 'localhost'
                    if 'MASTER_PORT' not in os.environ:
                        os.environ['MASTER_PORT'] = '12355'
                    if 'RANK' not in os.environ:
                        os.environ['RANK'] = '0'
                    if 'WORLD_SIZE' not in os.environ:
                        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
                    dist.init_process_group(backend='nccl', init_method='env://')
            except Exception:
                self.is_distributed = False
                self.rank = 0
                self.world_size = 1
        else:
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1
        
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        # Model with attention - use float32 to match baseline
        # Note: Using float32 instead of float16 to avoid dtype mismatches with DDP
        self.model = OptimizedKVCacheAttention().to(self.device)
        
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.model.eval()
        
        # Optimization: Pre-allocate KV cache (efficient management)
        batch_size = 2
        seq_len = 32
        hidden_dim = 256
        num_heads = 8
        head_dim = hidden_dim // num_heads
        
        # Pre-allocate cache with fixed size (efficient management) - use float32 to match model
        cache_size = (batch_size, num_heads, seq_len, head_dim)
        self.k_cache = torch.zeros(cache_size, device=self.device, dtype=torch.float32)
        self.v_cache = torch.zeros(cache_size, device=self.device, dtype=torch.float32)
        
        # Simulate autoregressive generation steps - use float32 to match baseline
        self.inputs = [
        torch.randn(batch_size, 1, hidden_dim, device=self.device, dtype=torch.float32)
            for _ in range(seq_len)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Efficient KV cache management."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_kv_cache_management", enable=enable_nvtx):
            with torch.no_grad():
                cache_pos = 0
                
        # Optimization: Efficient KV cache management
        # Pre-allocated cache, in-place updates, proper synchronization
        for step_input in self.inputs:
            pass
        output, self.k_cache, self.v_cache = self.model(
        step_input,
        self.k_cache,
        self.v_cache,
        cache_pos
        )
                    
    # Synchronize KV cache across GPUs (if distributed)
        if self.is_distributed:
            pass
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        output = output / self.world_size
                        
    # Efficient cache synchronization (only when needed)
    # In practice, would sync cache at specific intervals
                    
        cache_pos += step_input.shape[1]
                    
    # Optimization: Efficient management
    # - Pre-allocated cache (no reallocation)
    # - In-place updates (no copying)
    # - Bounded memory usage

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
        self.k_cache = None
        self.v_cache = None
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
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
        if self.inputs is None:
            return "Inputs not initialized"
        if self.k_cache is None or self.v_cache is None:
            return "KV cache not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedKVCacheManagementBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedKVCacheManagementBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: kv_cache_management")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
