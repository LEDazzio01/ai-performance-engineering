"""baseline_attention.py - Baseline SDPA without Flash/Flex kernels in advanced attention context.

Demonstrates scaled dot-product attention that disables Flash/FlexAcceleration.
Attention: Uses math-based SDP kernel which is significantly slower for long contexts.
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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class BaselineAttentionBenchmark(Benchmark):
    """Baseline: Math SDP kernel (Flash/Flex disabled)."""

    def __init__(self):
        self.device = resolve_device()
        self.qkv_proj = None
        self.input = None
        self.context_length = 4096
        self.hidden_dim = 512
        self.num_heads = 8
        self.batch_size = 4

    def setup(self) -> None:
        """Setup: Initialize projections and synthetic workload."""
        torch.manual_seed(42)
        self.qkv_proj = (
            nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=False)
            .to(device=self.device, dtype=torch.float16)
            .eval()
        )
        self.input = torch.randn(
            self.batch_size,
            self.context_length,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        torch.cuda.synchronize()

    def _matmul_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project activations into Q, K, V tensors."""
        qkv = self.qkv_proj(self.input)
        qkv = qkv.reshape(
            self.input.shape[0],
            self.context_length,
            3,
            self.num_heads,
            self.hidden_dim // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def benchmark_fn(self) -> None:
        """Benchmark: Execute SDPA with Flash kernels disabled."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("attention", enable=enable_nvtx):
            q, k, v = self._matmul_qkv()
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
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
        """Teardown: Clean up tensors."""
        self.qkv_proj = None
        self.input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
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
    """Factory function for benchmark discovery."""
    return BaselineAttentionBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Advanced Attention: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
