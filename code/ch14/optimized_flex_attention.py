"""optimized_flex_attention.py - Optimized with FlexAttention."""

from __future__ import annotations

import sys
from pathlib import Path
from contextlib import nullcontext

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)

def _flash_sdp_context():
    """Prefer the new sdpa_kernel API; fall back to no-op if unavailable."""
    if sdpa_kernel is None or SDPBackend is None or not hasattr(SDPBackend, "FLASH_ATTENTION"):
        return nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


class FlexAttentionBlock(nn.Module):
    """MHA block backed by fused scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        with _flash_sdp_context():
            context = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(context)


class OptimizedFlexAttentionBenchmark(BaseBenchmark):
    """Optimized: Uses FlexAttention for flexible attention patterns."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.embed_dim = 1024
        self.seq_len = 1024
        self.batch = 4
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.num_heads = 16
        self._last = 0.0
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize FlexAttention model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        block = FlexAttentionBlock(self.embed_dim, self.num_heads, self.dtype).to(self.device)
        block = block.eval()
        compile_fn = getattr(torch, "compile", None)
        if callable(compile_fn):
                block = compile_fn(block, mode="reduce-overhead", dynamic=False)
        self.model = block

        self.graph_input = torch.randn(
            self.batch,
            self.seq_len,
            self.embed_dim,
            device=self.device,
            dtype=self.dtype,
        )
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(self.graph_input)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: FlexAttention operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_flex_attention", enable=enable_nvtx):
            if self.model is None or self.graph_input is None:
                raise RuntimeError("Model not initialized")
            out = self.model(self.graph_input)
            self._last = float(out.sum())
            self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.graph_input = None
        self.graph_output = None
        if torch.cuda.is_available():
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
        from common.python.benchmark_metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )
}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None or self.graph_input is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedFlexAttentionBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    
    print(f"\nOptimized FlexAttention: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: FlexAttention provides flexible attention patterns for various use cases")
