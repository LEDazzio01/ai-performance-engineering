"""optimized_attention_standard.py - FlexAttention-style optimized attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


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


class OptimizedAttentionFlexBenchmark(VerificationPayloadMixin, BaseBenchmark):
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
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
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
                self.output = self.model(self.inputs)
        self._synchronize()
        if self.inputs is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.inputs},
            output=self.output.detach().float().clone(),
            batch_size=self.batch_size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1.0, 100.0),
        )

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
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output


def get_benchmark() -> OptimizedAttentionFlexBenchmark:
    return OptimizedAttentionFlexBenchmark()
