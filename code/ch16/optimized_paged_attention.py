"""optimized_paged_attention.py - Optimized paged attention in MoE context.

Demonstrates paged attention for efficient KV cache management.
Paged attention: Uses non-contiguous pages for efficient memory management.
Reduces fragmentation and improves memory utilization for variable-length sequences.
Implements BaseBenchmark for harness integration.
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
    BaseBenchmark,
    BenchmarkConfig,
)
from common.python.paged_attention import PagedKVCache, PagedAttentionConfig


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class OptimizedPagedAttentionBenchmark(BaseBenchmark):
    """Optimized: Paged attention for efficient KV cache management.
    
    Paged attention: Uses non-contiguous pages for efficient memory management.
    Reduces fragmentation and improves memory utilization for variable-length sequences.
    """
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model: Optional[nn.MultiheadAttention] = None
        self.cache_config: Optional[PagedAttentionConfig] = None
        self.kv_cache: Optional[PagedKVCache] = None
        self.prefill_inputs: Optional[torch.Tensor] = None
        self.decode_inputs: Optional[torch.Tensor] = None
        self.batch_size = 2
        self.prefill_len = 2048
        self.decode_steps = 512
        self.hidden_dim = 512
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
    
    def setup(self) -> None:
        """Setup: Initialize model and paged KV cache."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True
        ).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()
        
        self.cache_config = PagedAttentionConfig(
            batch_size=self.batch_size,
            page_size=32,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            device=self.device,
            dtype=self.model.in_proj_weight.dtype,
        )
        self.kv_cache = PagedKVCache(self.cache_config)

        model_dtype = self.model.in_proj_weight.dtype
        self.prefill_inputs = torch.randn(
            self.batch_size,
            self.prefill_len,
            self.hidden_dim,
            device=self.device,
            dtype=model_dtype,
        )
        self.decode_inputs = torch.randn(
            self.batch_size,
            self.decode_steps,
            self.hidden_dim,
            device=self.device,
            dtype=model_dtype,
        )
        total_tokens = self.batch_size * (self.prefill_len + self.decode_steps)
        self._synchronize()
        self.register_workload_metadata(
            tokens_per_iteration=float(total_tokens),
            requests_per_iteration=float(self.batch_size),
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Paged attention."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                if any(
                    item is None
                    for item in (
                        self.model,
                        self.prefill_inputs,
                        self.decode_inputs,
                        self.cache_config,
                    )
                ):
                    raise RuntimeError("Paged attention benchmark not initialized correctly")

                # Reset cache state for this run
                self.kv_cache = PagedKVCache(self.cache_config)  # type: ignore[arg-type]
                sequence = torch.cat(
                    [self.prefill_inputs, self.decode_inputs],
                    dim=1,
                )
                total_tokens = sequence.size(1)

                for pos in range(total_tokens):
                    token = sequence[:, pos : pos + 1, :]
                    qkv = F.linear(token, self.model.in_proj_weight, self.model.in_proj_bias)  # type: ignore[arg-type]
                    qkv = qkv.reshape(self.batch_size, 1, 3, self.num_heads, self.head_dim)
                    q, k, v = qkv.unbind(dim=2)

                    q_heads = q.permute(0, 2, 1, 3).contiguous()
                    self.kv_cache.write(pos, k.contiguous(), v.contiguous())  # type: ignore[arg-type]

                    k_all, v_all = self.kv_cache.get_kv(pos + 1)
                    k_heads = k_all.permute(0, 2, 1, 3).contiguous()
                    v_heads = v_all.permute(0, 2, 1, 3).contiguous()

                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q_heads,
                        k_heads,
                        v_heads,
                        is_causal=False,
                    )
                    _ = attn_output.sum()
        self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.kv_cache = None
        self.cache_config = None
        self.prefill_inputs = None
        self.decode_inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.prefill_inputs is None or self.decode_inputs is None:
            return "Inputs not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedPagedAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedPagedAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: paged_attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
