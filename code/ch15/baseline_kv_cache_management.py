"""baseline_kv_cache_management.py - Baseline KV cache without management."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineKVCacheManagementBenchmark(BaseBenchmark):
    """Baseline: recomputes KV every step (no cache reuse)."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.MultiheadAttention] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self.hidden_dim = 256
        self.num_heads = 8
        self.batch_size = 8
        self.steps = 32
        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.jitter_exemption_reason = "KV cache benchmark: fixed dimensions for cache comparison"
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Initialize model without KV cache management."""
        torch.manual_seed(42)
        self.model = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        ).to(self.device).eval()
        
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
            for _ in range(self.steps)
        ]
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: KV cache without management."""
        assert self.model is not None and self.inputs is not None
        with self._nvtx_range("baseline_kv_cache_management"):
            with torch.no_grad():
                for step, query in enumerate(self.inputs):
                    all_inputs = torch.cat(self.inputs[: step + 1], dim=1)
                    output, _ = self.model(query, all_inputs, all_inputs)
                    _ = output.sum()
            self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
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
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Inputs not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {"batch_size": self.batch_size, "steps": self.steps, "hidden_dim": self.hidden_dim}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineKVCacheManagementBenchmark()
