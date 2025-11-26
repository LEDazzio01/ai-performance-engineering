"""baseline_sdpa_attention.py - Naive multi-kernel attention (low arithmetic intensity).

Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity

This baseline demonstrates the naive approach to attention:
- Separate kernels for Q@K^T, softmax, and attn@V
- Multiple round-trips to global memory between operations
- Low arithmetic intensity due to unfused operations

The book mentions (line 164): "PyTorch's `scaled_dot_product_attention` (SDPA)
may dispatch to FlashAttention, memory-efficient, or cuDNN backends..."

This baseline intentionally uses separate operations to show why fusion matters.
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

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineSDPAAttentionBenchmark(BaseBenchmark):
    """Baseline: Naive multi-kernel attention without fusion.
    
    Demonstrates low arithmetic intensity due to:
    1. Separate matmul for Q @ K^T (writes intermediate to HBM)
    2. Separate softmax kernel (reads/writes HBM)
    3. Separate matmul for attn_weights @ V (reads from HBM)
    
    Each operation has low FLOPS/byte due to intermediate memory traffic.
    """

    def __init__(self):
        super().__init__()
        # Typical LLM attention dimensions
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
        """Naive attention: 3 separate kernels, 2 HBM round-trips."""
        with self._nvtx_range("baseline_sdpa_attention"):
            with torch.no_grad():
                # Kernel 1: Q @ K^T -> attn_scores (written to HBM)
                # Shape: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
                attn_scores = torch.matmul(
                    self.query, 
                    self.key.transpose(-2, -1)
                )
                
                # Scale (fused with matmul in optimized version)
                scale = 1.0 / (self.head_dim ** 0.5)
                attn_scores = attn_scores * scale
                
                # Kernel 2: Softmax (reads attn_scores from HBM, writes back)
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                # Kernel 3: attn_weights @ V -> output
                # Shape: [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
                output = torch.matmul(attn_weights, self.value)
                
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
        from common.python.benchmark_metrics import compute_roofline_metrics
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
    return BaselineSDPAAttentionBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselineSDPAAttentionBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline SDPA attention (unfused): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

