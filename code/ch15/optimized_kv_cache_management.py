"""optimized_kv_cache_management.py - Optimized KV cache management."""

from __future__ import annotations

from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


def _flash_sdp_context():
    """Prefer the new sdpa_kernel API; fall back to no-op if unavailable."""
    if sdpa_kernel is None or SDPBackend is None or not hasattr(SDPBackend, "FLASH_ATTENTION"):
        return nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


class OptimizedKVCacheManagementBenchmark(BaseBenchmark):
    """Optimized: KV cache reuse and management."""
    
    def __init__(self):
        super().__init__()
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self.cache_buffer: Optional[torch.Tensor] = None
        self.batch_size = 4
        self.hidden_dim = 256
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.steps = 32
        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Prefer flash; we'll verify compatibility below.
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
        torch.manual_seed(42)
        
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            module.eval()
        
        self.cache_buffer = torch.zeros(self.batch_size, self.steps, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
            for _ in range(self.steps)
        ]
        # Verify flash SDP works on this install/arch; otherwise skip.
        try:
            q = torch.randn(self.batch_size, 1, self.num_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            with _flash_sdp_context():
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: flash-attn kernels unavailable on this arch/install: {exc}") from exc
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None and self.out_proj is not None
        assert self.inputs is not None and self.cache_buffer is not None
        with self._nvtx_range("optimized_kv_cache_management"):
            with torch.no_grad():
                queries = torch.cat(self.inputs, dim=1)
                k_cache = self.cache_buffer.clone()
                
                q = self.q_proj(queries)
                k = self.k_proj(k_cache)
                v = self.v_proj(k_cache)
                
                q = q.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn = attn.transpose(1, 2).contiguous().view(self.batch_size, -1, self.hidden_dim)
                output = self.out_proj(attn)
                
                # Update cache with the newest token block without reallocation.
                self.cache_buffer.copy_(torch.cat([self.cache_buffer[:, 1:, :], queries[:, -1:, :]], dim=1))
                _ = output[:, -1, :].sum()
            self._synchronize()
    
    def teardown(self) -> None:
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.inputs = None
        self.cache_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
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
        if any(layer is None for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj)):
            return "Projection layers not initialized"
        if self.inputs is None or self.cache_buffer is None:
            return "Inputs/cache not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVCacheManagementBenchmark()
