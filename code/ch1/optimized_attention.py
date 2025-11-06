"""optimized attention - Optimized FlashAttention implementation. Implements Benchmark protocol for harness integration."""

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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedAttention(nn.Module):
    """Optimized multi-head attention using FlashAttention (scaled_dot_product_attention)."""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: optimized scaled dot-product attention (FlashAttention-style)."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, hidden)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Optimization: Use PyTorch's scaled_dot_product_attention (FlashAttention backend)
        # This uses FlashAttention when available, providing:
        # - O(N) memory complexity instead of O(N^2)
        # - Tiled computation to avoid storing full attention matrix
        # - Better performance for long sequences
        # Use is_causal=True for efficient causal masking
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True
        )  # (batch, heads, seq, head_dim)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output


class OptimizedAttentionBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.data = None
    
    def setup(self) -> None:
        """Setup: initialize optimized attention model and data."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        try:
            from common.python.compile_utils import compile_model
        except ImportError:
            compile_model = lambda m, **kwargs: m
        
        self.model = OptimizedAttention(hidden_dim=256, num_heads=8)
        
        if self.device.type == "cuda":
            try:
                self.model = self.model.to(self.device)
                # Optimization: Use FP16 for faster attention computation
                # FlashAttention benefits more from FP16 precision
                try:
                    self.model = self.model.half()
                    dtype = torch.float16
                except Exception:
                    dtype = torch.float32
                # Don't compile - FlashAttention is already optimized, compilation adds overhead
                # The key optimization is using scaled_dot_product_attention (FlashAttention backend)
            except Exception as exc:
                print(f"WARNING: GPU initialization failed: {exc}. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.cpu()
                dtype = torch.float32
        else:
            self.model = self.model.to(self.device)
            dtype = torch.float32
        
        self.model.eval()
        # Optimization: Use longer sequence length - FlashAttention benefits more from longer sequences
        # FlashAttention's O(N) memory complexity vs O(N^2) becomes more beneficial with longer sequences
        # Use FP16 for faster computation (FlashAttention optimized for FP16)
        seq_len = 256  # Longer sequence to show FlashAttention benefits
        self.data = torch.randn(4, seq_len, 256, device=self.device, dtype=dtype).contiguous()  # (batch, seq_len, hidden_dim)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_attention", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.data)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAttentionBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Attention (FlashAttention): {result.mean_ms:.3f} ms")
