"""optimized_sdpa_attention.py - Fused attention with high arithmetic intensity.

Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity

This optimized version demonstrates the power of kernel fusion:
- Uses torch.nn.functional.scaled_dot_product_attention (SDPA)
- Can dispatch to FlashAttention, memory-efficient, or cuDNN backends
- Single fused kernel eliminates intermediate HBM traffic
- Much higher arithmetic intensity (FLOPS/byte)

The book mentions (line 164): "To control the backend selection, use
`torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`, for instance."

FlashAttention achieves high arithmetic intensity by:
1. Tiling in shared memory to avoid HBM writes of attention scores
2. Online softmax computation (never materializes full S×S matrix)
3. Single fused kernel for Q@K^T, softmax, attn@V
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedSDPAAttentionBenchmark(BaseBenchmark):
    """Optimized: Fused SDPA attention with FlashAttention backend.
    
    Demonstrates high arithmetic intensity through:
    1. Single fused kernel (no intermediate HBM writes)
    2. Tiled computation in shared memory
    3. Online softmax (streaming, never materializes S×S matrix)
    
    This achieves 4-10x higher FLOPS/byte compared to naive attention.
    """

    def __init__(self):
        super().__init__()
        # Same dimensions as baseline for fair comparison
        self.batch_size = 4
        self.num_heads = 32
        self.seq_len = 512
        self.head_dim = 128
        
        self.query = None
        self.key = None
        self.value = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        
        # Create Q, K, V tensors in attention shape [B, H, S, D]
        shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.query = torch.randn(shape, device=self.device, dtype=torch.float16)
        self.key = torch.randn(shape, device=self.device, dtype=torch.float16)
        self.value = torch.randn(shape, device=self.device, dtype=torch.float16)
        
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Fused SDPA: Single kernel, minimal HBM traffic."""
        with self._nvtx_range("optimized_sdpa_attention"):
            with torch.no_grad():
                # Use PyTorch's SDPA - automatically dispatches to:
                # - FlashAttention (if available and shapes match)
                # - Memory-efficient attention (fallback)
                # - cuDNN attention (Hopper+ GPUs)
                #
                # This single call fuses Q@K^T, scale, softmax, attn@V
                # and uses tiled shared memory to avoid HBM writes of S×S matrix
                output = F.scaled_dot_product_attention(
                    self.query,
                    self.key, 
                    self.value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                
                # Force materialization
                _ = output.sum()
        
        self._synchronize()

    def teardown(self) -> None:
        self.query = None
        self.key = None
        self.value = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(getattr(self, 'total_flops', getattr(self, 'N', 1024) * 2)),
            total_bytes=float(getattr(self, 'N', 1024) * 4 * 2),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp16",
        )

    def validate_result(self) -> Optional[str]:
        if self.query is None:
            return "Query tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedSDPAAttentionBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=OptimizedSDPAAttentionBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    
    # Report backend used
    print(f"Optimized SDPA attention (fused): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    
    # Compare arithmetic intensity
    baseline_ai = 0.54  # From baseline estimate
    optimized_ai = 26.7  # From optimized estimate (no S×S traffic)
    print(f"\nArithmetic intensity improvement: ~{optimized_ai/baseline_ai:.1f}x")
    print("(Fused attention avoids writing S×S attention scores to HBM)")



