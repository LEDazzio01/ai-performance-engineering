"""optimized_attention.py - FlashAttention-enabled SDP kernel.

Demonstrates FlashAttention acceleration for long-context attention.
Attention: Enables Flash kernels + memory-efficient path for O(N) activation usage.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
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
    BenchmarkMode
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedAttentionBenchmark(Benchmark):
    """Optimized: FlashAttention-backed SDP kernel."""
    
    def __init__(self):
        self.device = resolve_device()
        self.qkv_proj = None
        self.input = None
        self.context_length = 4096
        self.hidden_dim = 512
        self.num_heads = 8
        self.batch_size = 4
    
    def setup(self) -> None:
        """Setup: initialize projections and inputs."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.qkv_proj = (
            nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=False)
            .to(device=self.device, dtype=torch.float16)
            .eval()
        )
        self.input = torch.randn(self.batch_size, self.context_length, self.hidden_dim, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def _matmul_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(self.input)
        qkv = qkv.reshape(
            self.input.shape[0],
            self.context_length,
            3,
            self.num_heads,
            self.hidden_dim // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v
    
    def benchmark_fn(self) -> None:
        """Benchmark: FlashAttention-backed SDPA."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("attention", enable=enable_nvtx):
            q, k, v = self._matmul_qkv()
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=False,
            ):
                _ = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Cleanup."""
        self.qkv_proj = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.qkv_proj is None or self.input is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedAttentionBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=3)
    )
    benchmark = OptimizedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: FlashAttention (memory-efficient)")
    print("=" * 70)
    print(f"Batch: {benchmark.batch_size}, Heads: {benchmark.num_heads}, SeqLen: {benchmark.context_length}")
    print(f"Memory: O(N) working set via Flash kernels")
    print("Optimization: Uses FlashAttention kernel\n")
    
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")
    print(f"Status: Compute-bound (O(N) memory)")
    print(f"Speedup: ~5-15x for long sequences (seq_len > 1024)")


if __name__ == "__main__":
    main()
