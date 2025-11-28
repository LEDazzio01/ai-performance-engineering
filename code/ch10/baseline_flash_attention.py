"""baseline_flash_attention.py - Baseline attention without FlashAttention in GEMM context."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineFlashAttentionBenchmark(BaseBenchmark):
    """Baseline: Standard attention without FlashAttention optimizations.
    
    Uses manual attention computation (matmul + softmax + matmul) which
    materializes the full O(seq_len²) attention matrix.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input = None
        self.batch_size = 4
        self.seq_len = 1024  # Match optimized version
        self.hidden_dim = 512  # Match optimized version
        self.num_heads = 8  # Match optimized version
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize attention model without FlashAttention."""
        torch.manual_seed(42)
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device)
        
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Baseline: FP32 input (no tensor core acceleration)
        self.input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.float32)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                self._manual_attention(self.input)
        torch.cuda.synchronize(self.device)
    
    def _manual_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Manual attention that materializes full attention matrix."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Manual attention: Q @ K^T -> O(seq_len²) memory
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # attn_weights @ V
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard attention without FlashAttention."""
        with self._nvtx_range("baseline_flash_attention"):
            with torch.no_grad():
                _output = self._manual_attention(self.input)
            self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
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
        if self.q_proj is None:
            return "Projections not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineFlashAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Flash Attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
