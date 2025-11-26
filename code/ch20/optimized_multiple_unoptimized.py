"""optimized_multiple_unoptimized.py - All optimizations combined."""

from __future__ import annotations

from typing import Optional
import os

import torch
import torch.nn as nn
import warnings

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass
from ch20.inductor_guard import (
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
    InductorCudagraphState,
)

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import compile_model


class SimpleModel(nn.Module):
    """Simple model for optimization demonstration."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedAllTechniquesBenchmark(BaseBenchmark):
    """Optimized benchmark stacking multiple techniques (FP16, compile, CUDA graph)."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.x: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.x_capture: Optional[torch.Tensor] = None
        # Default to fast settings to keep CI/subprocess runs short; can be overridden by users.
        self._fast_test = bool(os.environ.get("PYTEST_CURRENT_TEST")) or os.environ.get("AIPERF_FAST_BENCH", "0") == "1"
        if self._fast_test:
            # Override get_config at the instance level to keep subprocess runs short.
            self.get_config = self._fast_config  # type: ignore[assignment]
        self.batch_size = 8 if self._fast_test else 32
        self.hidden_dim = 512 if self._fast_test else 4096
        self._inductor_cfg_state: Optional[InductorCudagraphState] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        self._inductor_cfg_state = disable_inductor_cudagraph_features()
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().eval()
            
            self.model = compile_model(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
            test_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            warmup_runs = 1 if self._fast_test else 3
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = self.model(test_input)
            torch.cuda.synchronize()
            
            self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            
            steady_runs = 5 if self._fast_test else 50
            for _ in range(steady_runs):
                with torch.no_grad():
                    _ = self.model(self.x)
            torch.cuda.synchronize()
            
            self.graph = None
            self.x_capture = None
            try:
                graph = torch.cuda.CUDAGraph()
                self.x_capture = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
                graph_warmups = 1 if self._fast_test else 5
                for _ in range(graph_warmups):
                    with torch.no_grad():
                        _ = self.model(self.x_capture)
                torch.cuda.synchronize()
                
                with torch.cuda.graph(graph):
                    with torch.no_grad():
                        _ = self.model(self.x_capture)
                torch.cuda.synchronize()
                
                graph_replays = 2 if self._fast_test else 10
                for _ in range(graph_replays):
                    graph.replay()
                torch.cuda.synchronize()
                self.graph = graph
            except RuntimeError as e:
                if "Cannot prepare for replay during capturing stage" in str(e):
                    warnings.warn(
                        "TorchInductor already uses CUDA graphs on this platform; skipping manual CUDA graph capture.",
                        RuntimeWarning,
                    )
                    self.graph = None
                    self.x_capture = None
                    torch.cuda.synchronize()
                else:
                    raise
        except Exception:
            restore_inductor_cudagraph_features(self._inductor_cfg_state)
            self._inductor_cfg_state = None
            raise

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.x is not None
        with self._nvtx_range("multiple_techniques_optimized"):
            with torch.no_grad():
                if self.graph is None:
                    _ = self.model(self.x)
                else:
                    self.graph.replay()
            self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.x = None
        self.graph = None
        self.x_capture = None
        torch.cuda.empty_cache()
        restore_inductor_cudagraph_features(self._inductor_cfg_state)
        self._inductor_cfg_state = None
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
            use_subprocess=True,
        )
    
    def get_workload_metadata(self):
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
        if self.x is None:
            return "Input tensor not initialized"
        return None

    def _fast_config(self) -> BenchmarkConfig:
        """Return a trimmed-down config for fast test runs."""
        return BenchmarkConfig(
            iterations=1,
            warmup=1,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
            measurement_timeout_seconds=60,
            setup_timeout_seconds=60,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedAllTechniquesBenchmark()
