"""optimized_flash_attention.py - FlashAttention via SDPA demonstrating tiled attention.

This module demonstrates how FlashAttention achieves O(seq_len) memory complexity
through the same intra-kernel pipelining and tiling concepts taught in Chapter 10.

FlashAttention's key insight (aligning with Ch10 concepts):
- Standard attention materializes the full [seq_len × seq_len] attention matrix
- FlashAttention tiles the computation, processing attention in SRAM blocks
- This is exactly the producer-consumer pipelining pattern from this chapter
- Memory: O(seq_len) instead of O(seq_len²) by never materializing full matrix

The optimization here is selecting the SDPA backend that uses tiled attention,
demonstrating the practical benefit of the intra-kernel pipelining concepts.
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
from contextlib import contextmanager, nullcontext
from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)

# Use new SDPA API when available (PyTorch 2.2+)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _NEW_SDPA_API = True
except ImportError:
    sdpa_kernel = None  # type: ignore[assignment]
    SDPBackend = None  # type: ignore[assignment]
    _NEW_SDPA_API = False


@contextmanager
def sdpa_flash_backend():
    """Context manager to select FlashAttention backend in SDPA.
    
    FlashAttention uses the same principles taught in Chapter 10:
    - Tiled computation: processes attention in blocks that fit in SRAM
    - Pipelining: overlaps memory loads with softmax/matmul computation  
    - Never materializes the full O(seq_len²) attention matrix
    
    This is analogous to the double-buffered pipeline pattern but applied
    to the attention computation itself.
    """
    # Use new SDPA API when available (PyTorch 2.2+)
    if _NEW_SDPA_API and sdpa_kernel is not None:
        # Check which backends are available and select the best one
        backends = []
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            # Flash attention typically needs SM 8.0+ (Ampere), may have issues on SM 10.0
            if major >= 10:
                # On Blackwell (SM 10.0), prefer memory-efficient which is more stable
                backends = [SDPBackend.EFFICIENT_ATTENTION]
            elif major < 8:
                # Older GPUs don't support flash
                backends = [SDPBackend.EFFICIENT_ATTENTION]
            else:
                backends = [SDPBackend.FLASH_ATTENTION]
        else:
            backends = [SDPBackend.FLASH_ATTENTION]
        
        with sdpa_kernel(backends):
            yield
    else:
        # No context manager available, just yield
        yield


class TiledAttentionModule(nn.Module):
    """Attention module using tiled computation via SDPA.
    
    This demonstrates the practical application of Chapter 10's concepts:
    - The attention computation is broken into tiles
    - Each tile fits in shared memory (like our GEMM examples)
    - Softmax is computed incrementally to avoid storing full matrix
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # Separate Q, K, V projections (could fuse, but keeping simple for ch10)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """Forward pass using tiled attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            is_causal: If True, apply causal mask (autoregressive)
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Tiled attention via SDPA
        # Internally, this uses the same tiling strategy discussed in Ch10:
        # - Break Q, K, V into tiles that fit in SRAM
        # - Compute partial attention scores per tile
        # - Accumulate softmax incrementally (online softmax)
        # - Never store the full [seq_len × seq_len] attention matrix
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        # Reshape back: [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(attn_output)


class OptimizedFlashAttentionBenchmark(BaseBenchmark):
    """Optimized: Tiled attention via FlashAttention SDPA backend.
    
    This benchmark demonstrates the practical benefit of Chapter 10's
    intra-kernel pipelining concepts applied to attention:
    
    Baseline (standard attention):
    - Computes full [seq_len × seq_len] attention matrix
    - Memory: O(seq_len²) 
    - For seq_len=4096: 4096² × 4 bytes × batch × heads = HUGE
    
    Optimized (FlashAttention/tiled):
    - Tiles the computation, never stores full matrix
    - Memory: O(seq_len)
    - Uses same producer-consumer pattern as Ch10's pipeline examples
    
    Expected improvement: Memory usage scales linearly, not quadratically.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        
        # Use larger sizes to show the memory benefit of tiling
        self.batch_size = 4
        self.seq_len = 1024  # At 1024, O(n²) vs O(n) matters significantly
        self.hidden_dim = 512
        self.num_heads = 8
        self.use_causal = True
        
        self.input: Optional[torch.Tensor] = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tiled attention model."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            enable_tf32()  # Enable TF32 for Tensor Cores (Ampere+)
        
        torch.manual_seed(42)
        
        # Use FP16 for tensor core acceleration
        self.model = TiledAttentionModule(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).to(self.device).half().eval()
        
        # Input tensor in FP16
        self.input = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=torch.float16
        )
        
        # Warmup with tiled backend
        with torch.no_grad(), sdpa_flash_backend():
            for _ in range(3):
                _ = self.model(self.input, is_causal=self.use_causal)
        
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tiled attention computation."""
        with self._nvtx_range("optimized_tiled_attention"):
            with torch.no_grad(), sdpa_flash_backend():
                # The SDPA call internally uses tiled computation
                # This is the same pipelining concept from Ch10 applied to attention
                _output = self.model(self.input, is_causal=self.use_causal)
        
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        
        # Verify tiled attention produces valid output
        with torch.no_grad(), sdpa_flash_backend():
            output = self.model(self.input[:1], is_causal=False)
            if torch.isnan(output).any():
                return "NaN values in attention output"
        
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedFlashAttentionBenchmark()


def main() -> None:
    """Standalone execution with explanation of Ch10 concepts."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=10)
    )
    benchmark = OptimizedFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Tiled Attention (FlashAttention via SDPA)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Batch size: {benchmark.batch_size}")
    print(f"  Sequence length: {benchmark.seq_len}")
    print(f"  Hidden dim: {benchmark.hidden_dim}")
    print(f"  Num heads: {benchmark.num_heads}")
    print()
    print(f"Results:")
    print(f"  Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print()
    print("Chapter 10 concepts demonstrated:")
    print("  - Tiled computation (attention computed in SRAM-sized blocks)")
    print("  - Producer-consumer pattern (load K/V tiles, compute partial scores)")
    print("  - Memory efficiency: O(seq_len) instead of O(seq_len²)")
    print()
    print("Memory comparison at seq_len=1024:")
    print(f"  Standard attention: {benchmark.seq_len**2 * 4 / 1024:.0f} KB per head")
    print(f"  Tiled attention: ~{benchmark.seq_len * 4 / 1024:.0f} KB per head")


if __name__ == "__main__":
    main()
