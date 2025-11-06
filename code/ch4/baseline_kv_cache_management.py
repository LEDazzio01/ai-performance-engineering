"""baseline_kv_cache_management.py - Baseline KV cache without management in multi-GPU.

Demonstrates KV cache usage without proper management across GPUs.
KV cache management: This baseline does not implement efficient KV cache management.
Recomputes or inefficiently manages cache across distributed GPUs.
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

from common.python.compile_utils import compile_model

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
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
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


class BaselineKVCacheManagementBenchmark(Benchmark):
    """Baseline: KV cache without efficient management across GPUs.
    
    KV cache management: This baseline does not implement efficient KV cache management.
    Cache is recomputed or inefficiently managed, causing memory overhead.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self) -> None:
        """Setup: Initialize model and inputs."""
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
                self.is_distributed = True
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            except Exception:
                self.is_distributed = False
        
        # Model with attention
        self.model = BaselineKVCacheAttention().to(self.device)
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.model.eval()
        
        # Simulate autoregressive generation steps
        batch_size = 2
        seq_len = 32
        hidden_dim = 256
        self.inputs = [
            torch.randn(batch_size, 1, hidden_dim, device=self.device)
            for _ in range(seq_len)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: KV cache without efficient management."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_kv_cache_management", enable=enable_nvtx):
            with torch.no_grad():
                kv_cache = None
                
                # Baseline: No efficient KV cache management
                # Cache is inefficiently managed (reallocated each step)
                for step_input in self.inputs:
                    output, kv_cache = self.model(step_input, kv_cache)
                    
                    # Synchronize across GPUs (if distributed)
                    if self.is_distributed:
                        dist.all_reduce(output, op=dist.ReduceOp.SUM)
                        output = output / self.world_size
                    
                    # Baseline: No efficient cache management
                    # Cache grows unbounded, causing memory overhead

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
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
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineKVCacheManagementBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineKVCacheManagementBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: kv_cache_management")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
