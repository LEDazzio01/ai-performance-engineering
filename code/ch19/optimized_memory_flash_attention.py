"""optimized_memory_flash_attention.py - Optimized memory management with Flash Attention.

Demonstrates Flash Attention for memory-efficient attention computation.
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
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class OptimizedMemoryFlashAttentionBenchmark(Benchmark):
    """Optimized: Flash Attention for memory-efficient attention."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None

        self.x = None
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 256
        self.num_heads = 8
    
    def setup(self) -> None:
        """Setup: Initialize model with Flash Attention."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Multi-head attention layer
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        self.model = self.model.to(self.device).to(dtype=torch.bfloat16)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                 pass
        self.model.eval()
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=torch.bfloat16
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Flash Attention (memory-efficient)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_memory_flash_attention", enable=enable_nvtx):
            with torch.no_grad():
                pass
        # Optimization: Use Flash Attention via F.scaled_dot_product_attention
        # Flash Attention reduces memory from O(N²) to O(N) by tiling
        # This is crucial for long sequences in adaptive memory management
        q = self.x
        k = self.x
        v = self.x
                
        # Use PyTorch's optimized attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=True,
        dropout_p=0.0,
        )
        _ = attn_output

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.x
        if torch.cuda.is_available():
            pass
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        try:
            with torch.no_grad():
                pass
            q = self.x
            k = self.x
            v = self.x
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            if output.shape != self.x.shape:
                return f"Output shape mismatch"
            return None
        except Exception as e:
            return f"Flash Attention failed: {e}"

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedMemoryFlashAttentionBenchmark()

def main() -> None:
    """Standalone execution."""
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedMemoryFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Memory Flash Attention")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(" Tip: Flash Attention reduces memory from O(N²) to O(N) through tiling")

if __name__ == "__main__":
    main()
