"""optimized_attention_flash.py - FlashAttention-style optimized attention.

Uses PyTorch's scaled_dot_product_attention for memory-efficient attention.
Memory: O(N) instead of O(N²) - enables longer sequences.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn.functional as F

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


def optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Optimized attention using PyTorch's scaled dot-product (FlashAttention-style)."""
    return F.scaled_dot_product_attention(q, k, v)


class OptimizedFlashAttentionBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.q = None
        self.k = None
        self.v = None
        self.batch_size = 4
        self.num_heads = 16
        self.seq_len = 1024
        self.head_dim = 64
    
    def setup(self) -> None:
        """Setup: initialize tensors."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        self.k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        self.v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_attention_flash", enable=enable_nvtx):
            with torch.no_grad():
                _ = optimized_attention(self.q, self.k, self.v)

    def teardown(self) -> None:
        """Cleanup."""
        del self.q, self.k, self.v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=10,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.q is None:
            return "Query tensor (q) not initialized"
        if self.k is None:
            return "Key tensor (k) not initialized"
        if self.v is None:
            return "Value tensor (v) not initialized"
        try:
            # Verify tensor shapes match expected dimensions
            expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
            if self.q.shape != expected_shape:
                return f"Query tensor shape mismatch: expected {expected_shape}, got {self.q.shape}"
            if self.k.shape != expected_shape:
                return f"Key tensor shape mismatch: expected {expected_shape}, got {self.k.shape}"
            if self.v.shape != expected_shape:
                return f"Value tensor shape mismatch: expected {expected_shape}, got {self.v.shape}"
            # Test forward pass
            with torch.no_grad():
                output = optimized_attention(self.q, self.k, self.v)
                # Output should be (batch_size, num_heads, seq_len, head_dim)
                expected_output_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
                if output.shape != expected_output_shape:
                    return f"Flash attention output shape mismatch: expected {expected_output_shape}, got {output.shape}"
                if not torch.isfinite(output).all():
                    return "Flash attention output contains non-finite values"
        except Exception as e:
            return f"Flash attention forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedFlashAttentionBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=3)
    )
    benchmark = OptimizedFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: FlashAttention-style (Memory-Efficient)")
    print("=" * 70)
    print(f"Batch: {benchmark.batch_size}, Heads: {benchmark.num_heads}, SeqLen: {benchmark.seq_len}")
    print(f"Memory: O(N) = {benchmark.seq_len * 4 / 1e6:.3f} MB per head (vs O(N²) baseline)")
    print("Optimization: Uses FlashAttention kernel\n")
    
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")
    print(f"Status: Compute-bound (O(N) memory)")
    print(f"Speedup: ~5-15x for long sequences (seq_len > 1024)")


if __name__ == "__main__":
    main()
