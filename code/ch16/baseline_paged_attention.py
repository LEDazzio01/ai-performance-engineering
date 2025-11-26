"""baseline_paged_attention.py - Attention without paged KV caches."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class BaselinePagedAttentionBenchmark(BaseBenchmark):
    """Baseline: contiguous KV cache (no paging)."""

    def __init__(self):
        super().__init__()
        self.batch_size = 2
        self.hidden_dim = 512
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.max_seq_len = 2048
        self.steps = 512
        self.qkv_proj: Optional[nn.Linear] = None
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.inputs: Optional[torch.Tensor] = None
        tokens = self.batch_size * (self.max_seq_len + self.steps)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        # Keep input and projection weights in the same dtype to avoid matmul dtype mismatches
        self.qkv_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim * 3,
            bias=True,
            device=self.device,
            dtype=torch.float16,
        )
        self.k_cache = torch.empty(
            self.batch_size,
            self.max_seq_len,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.v_cache = torch.empty_like(self.k_cache)
        self.inputs = torch.randn(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.qkv_proj is None or self.k_cache is None or self.v_cache is None or self.inputs is None:
            raise RuntimeError("Model or caches not initialized")
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_paged_attention", enable=enable_nvtx):
            with torch.no_grad():
                qkv = self.qkv_proj(self.inputs)
                q, k, v = torch.chunk(qkv, 3, dim=-1)
                q = q.view(self.batch_size, self.max_seq_len, self.num_heads, self.head_dim)
                k = k.view_as(torch.empty_like(self.k_cache))
                v = v.view_as(torch.empty_like(self.v_cache))
                self.k_cache.copy_(k)
                self.v_cache.copy_(v)

                attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, self.k_cache)
                attn_scores = attn_scores / (self.head_dim ** 0.5)
                attn_probs = F.softmax(attn_scores, dim=-1)
                _ = torch.einsum("bhqk,bkhd->bqhd", attn_probs, self.v_cache)
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.qkv_proj = None
        self.k_cache = None
        self.v_cache = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

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
        if self.k_cache is None or self.v_cache is None:
            return "KV caches not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselinePagedAttentionBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselinePagedAttentionBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline paged attention: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
