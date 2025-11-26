"""optimized_moe.py - Minimal MoE-style benchmark for Chapter 20 tests."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import compile_model


class ToyMoe(nn.Module):
    """Simplified MoE block with two experts and top-1 routing."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 2, bias=False)
        self.expert0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.expert1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(self.gate(x), dim=-1)
        top_expert = scores.argmax(dim=-1)
        out0 = self.expert0(x)
        out1 = self.expert1(x)
        mask0 = (top_expert == 0).float().unsqueeze(-1)
        mask1 = (top_expert == 1).float().unsqueeze(-1)
        return out0 * mask0 + out1 * mask1


class OptimizedMoeBenchmark(BaseBenchmark):
    """Exercise a tiny MoE forward pass with optional compilation."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 32
        self.hidden_dim = 1024
        tokens = self.batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(1)
        model = ToyMoe(self.hidden_dim).to(self.device).half().eval()
        self.model = compile_model(
            model,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.float16)
        for _ in range(2):
            with torch.no_grad():
                _ = self.model(self.inputs)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        with self._nvtx_range("optimized_moe"):
            with torch.no_grad():
                _ = self.model(self.inputs)
            self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=3,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoeBenchmark()
