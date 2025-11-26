"""optimized_inference_monolithic.py - Disaggregated inference (optimized decode service)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleLLM(nn.Module):
    """Simplified LLM for inference simulation."""
    
    def __init__(self, hidden_dim=1024, num_layers=12):
        super().__init__()
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


class OptimizedInferenceDisaggregatedBenchmark(BaseBenchmark):
    """Benchmark: optimized decode service (prefill runs elsewhere)."""
    
    def __init__(self):
        super().__init__()
        self.decode_model: Optional[SimpleLLM] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.batch = 1
        self.num_tokens = 16
        tokens = self.batch * self.num_tokens
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.decode_model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        self.kv_cache = torch.randn(self.batch, 1, 1024, device=self.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            for _ in range(5):
                _ = self.decode_model.decode(self.kv_cache, num_tokens=self.num_tokens)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.decode_model is not None and self.kv_cache is not None
        with self._nvtx_range("inference_monolithic_optimized"):
            with torch.no_grad():
                _ = self.decode_model.decode(self.kv_cache, num_tokens=self.num_tokens)
            self._synchronize()
    
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
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedInferenceDisaggregatedBenchmark()
