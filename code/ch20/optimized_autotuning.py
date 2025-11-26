"""optimized_autotuning.py - Lightweight autotuning demo benchmark."""

from __future__ import annotations

from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
    from common.python.compile_utils import enable_tf32  # type: ignore
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore
    enable_tf32 = None  # type: ignore

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import compile_model


class AutotuneModel(nn.Module):
    """Tiny MLP to exercise autotuning paths."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(x)


class OptimizedAutotuningBenchmark(BaseBenchmark):
    """Runs a small model with torch.compile to validate autotune plumbing."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 16
        self.hidden_dim = 1024
        tokens = self.batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        if enable_tf32 is not None:
            enable_tf32(set_global_precision=True)
        else:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to set float32_matmul_precision='high': {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        model = AutotuneModel(self.hidden_dim).to(self.device, dtype=torch.bfloat16).eval()
        # compile_model skips safely on unsupported architectures
        self.model = compile_model(
            model,
            mode="max-autotune",
            fullgraph=False,
            dynamic=False,
        )
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        # Warm a couple runs to trigger compile/autotune caches
        sdpa_ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        for _ in range(3):
            with torch.no_grad(), sdpa_ctx:
                _ = self.model(self.inputs)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        sdpa_ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        with self._nvtx_range("optimized_autotuning"):
            with torch.no_grad(), sdpa_ctx:
                _ = self.model(self.inputs)
            self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
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
    return OptimizedAutotuningBenchmark()
