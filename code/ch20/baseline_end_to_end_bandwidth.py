"""baseline_end_to_end_bandwidth.py - Baseline end-to-end bandwidth (sequential)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimplePipeline(nn.Module):
    """Simple inference pipeline."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineEndToEndBandwidthBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline end-to-end bandwidth - sequential processing."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self.outputs: Optional[list[torch.Tensor]] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self.num_batches = 10
        tokens = self.batch_size * self.num_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size * self.num_batches),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = SimplePipeline(hidden_dim=self.hidden_dim).to(self.device, dtype=torch.float32).eval()
        self.inputs = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
            for _ in range(self.num_batches)
        ]
        self.outputs = []
        self.output = None
        for inp in self.inputs[:3]:
            _ = self.model(inp)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        with self._nvtx_range("baseline_end_to_end_bandwidth"):
            self.outputs = []
            with torch.no_grad():
                for inp in self.inputs:
                    out = self.model(inp)
                    self.outputs.append(out)
            if self.outputs:
                self.output = torch.stack(self.outputs)
            else:
                self.output = None
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.model is None or self.inputs is None or self.output is None:
            raise RuntimeError("capture_verification_payload() requires completed benchmark run")
        stacked_inputs = torch.stack(self.inputs)
        self._set_verification_payload(
            inputs={"inputs": stacked_inputs},
            output=self.output,
            batch_size=int(stacked_inputs.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.outputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.outputs is None or len(self.outputs) != self.num_batches:
            return f"Expected {self.num_batches} outputs"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()


def get_benchmark() -> BaseBenchmark:
    return BaselineEndToEndBandwidthBenchmark()
