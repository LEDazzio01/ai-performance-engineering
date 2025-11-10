"""baseline_paged_attention.py - Baseline attention without paged attention.

Demonstrates attention computation without paged attention optimization.
Paged attention: This baseline does not use paged attention for KV cache management.
Uses contiguous memory allocation, causing fragmentation and inefficiency.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselinePagedAttentionBenchmark(Benchmark):
    """Baseline: Attention without paged attention.
    
    Paged attention: This baseline does not use paged attention for KV cache management.
    Uses contiguous memory allocation, causing fragmentation and inefficient memory usage.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.hidden_dim = 512
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.batch_size = 4
        self.max_seq_len = 256
        self.steps = 512
    
    def setup(self) -> None:
        """Setup: Initialize model and contiguous KV cache."""
        torch.manual_seed(42)
        # Baseline: No paged attention - contiguous memory allocation
        # Paged attention uses non-contiguous pages for efficient memory management
        # This baseline allocates contiguous memory for each sequence
        
        # Simple attention model
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        ).to(self.device).eval()
        
        # Simulate autoregressive generation
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
            for _ in range(self.steps)
        ]
        torch.cuda.synchronize()

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, hidden) into (B, heads, T, head_dim)."""
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _project_qkv(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs into Q, K, V using the MHA weights."""
        qkv = F.linear(query, self.model.in_proj_weight, self.model.in_proj_bias)
        return qkv.chunk(3, dim=-1)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention without paged attention."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                k_history: List[torch.Tensor] = []
                v_history: List[torch.Tensor] = []
                for step, query in enumerate(self.inputs):
                    q_proj, k_proj, v_proj = self._project_qkv(query)
                    q_heads = self._split_heads(q_proj)  # (B, H, 1, Dh)
                    k_new = self._split_heads(k_proj).permute(0, 2, 1, 3)  # (B, 1, H, Dh)
                    v_new = self._split_heads(v_proj).permute(0, 2, 1, 3)
                    
                    # Naive baseline: reallocate storage via torch.cat each step
                    k_history.append(k_new.clone())
                    v_history.append(v_new.clone())
                    
                    k_all = torch.cat(k_history, dim=1)  # (B, T, H, Dh)
                    v_all = torch.cat(v_history, dim=1)
                    k_heads = k_all.permute(0, 2, 1, 3).contiguous()  # (B, H, T, Dh)
                    v_heads = v_all.permute(0, 2, 1, 3).contiguous()
                    
                    # Scaled dot-product attention without paging
                    attn_scores = torch.matmul(
                        q_heads, k_heads.transpose(-2, -1)
                    ) / math.sqrt(self.head_dim)
                    attn_probs = torch.softmax(attn_scores, dim=-1)
                    context = torch.matmul(attn_probs, v_heads)  # (B, H, 1, Dh)
                    
                    # Merge heads (baseline output not used further, we just ensure work is done)
                    _ = context.permute(0, 2, 1, 3).reshape(query.size(0), 1, self.hidden_dim)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
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
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselinePagedAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselinePagedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: paged_attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
