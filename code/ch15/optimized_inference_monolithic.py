"""optimized_inference_monolithic.py - Disaggregated inference (optimized decode service)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


class SimpleLLM(nn.Module):
    """Simplified LLM for inference simulation."""
    
    def __init__(self, hidden_dim=1024, num_layers=12):
        super().__init__()
        self.output = None
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        )
    
    def decode(self, kv_cache, num_tokens=16):
        """Decode: generate tokens (memory-bound)."""
        outputs = []
        x = kv_cache
        for _ in range(num_tokens):
            for layer in self.layers:
                x = torch.relu(layer(x))
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class OptimizedInferenceDisaggregatedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark: optimized decode service (prefill runs elsewhere)."""
    
    def __init__(self):
        super().__init__()
        self.decode_model: Optional[SimpleLLM] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch = 1
        self.num_tokens = 16
        tokens = self.batch * self.num_tokens
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_kv: Optional[torch.Tensor] = None
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.decode_model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        self.kv_cache = torch.randn(self.batch, 1, 1024, device=self.device, dtype=torch.bfloat16)
        self.output = None
        self._synchronize()
        self._verify_kv = self.kv_cache.detach()
    
    def benchmark_fn(self) -> None:
        assert self.decode_model is not None and self.kv_cache is not None
        with self._nvtx_range("inference_monolithic_optimized"):
            with torch.no_grad():
                self.output = self.decode_model.decode(self.kv_cache, num_tokens=self.num_tokens)

    def capture_verification_payload(self) -> None:
        if self.decode_model is None or self.kv_cache is None or self.output is None or self._verify_kv is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"kv": self._verify_kv},
            output=self.output,
            batch_size=int(self.kv_cache.shape[0]),
            parameter_count=sum(p.numel() for p in self.decode_model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )
    
    def teardown(self) -> None:
        self.decode_model = None
        self.kv_cache = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self):
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedInferenceDisaggregatedBenchmark()
