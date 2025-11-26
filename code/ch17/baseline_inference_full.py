"""baseline_inference_full.py - Full-depth inference without early exit."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class FullDepthModel(nn.Module):
    def __init__(self, hidden_dim: int = 2048, num_layers: int = 24):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


class BaselineInferenceFullBenchmark(BaseBenchmark):
    """Always executes every layer (no early exit)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch_size = 16
        self.hidden_dim = 2048
        self.num_layers = 24
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)

        self.model = FullDepthModel(self.hidden_dim, self.num_layers).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()

        input_dtype = next(self.model.parameters()).dtype
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=input_dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None

        with self._nvtx_range("inference_full"):
            with torch.no_grad():
                _ = self.model(self.inputs)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

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
        if self.model is None or self.inputs is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> BaselineInferenceFullBenchmark:
    return BaselineInferenceFullBenchmark()
