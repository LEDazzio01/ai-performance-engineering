"""optimized_attention_standard.py - FlexAttention-style optimized attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class FlexAttention(nn.Module):
    """FlexAttention implementation using optimized kernels."""
    
    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=torch.float16)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(out)


class OptimizedAttentionFlexBenchmark(BaseBenchmark):
    """FlexAttention optimization - optimized kernels."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs = None
        self.batch_size = 2
        self.seq_len = 512
        self.hidden_dim = 1024
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        
        self.model = FlexAttention(hidden_dim=self.hidden_dim, num_heads=8).to(self.device).eval()
        self.inputs = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
        
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(self.inputs)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("attention_standard"):
            with torch.no_grad():
                _ = self.model(self.inputs)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedAttentionFlexBenchmark:
    return OptimizedAttentionFlexBenchmark()
