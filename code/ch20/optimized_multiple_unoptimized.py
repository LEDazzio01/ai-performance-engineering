"""optimized_multiple_unoptimized.py - All optimizations combined.

Chapter 20 demonstrates stacking multiple optimizations:
1. FP16 precision for tensor core acceleration
2. torch.compile for kernel fusion
3. CUDA graphs for reduced launch overhead
"""

from __future__ import annotations

from typing import Optional
import os
import warnings

import torch
import torch.nn as nn

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
        # Larger batch size to amortize compile/graph overhead
        self.batch_size = 128
        self.hidden_dim = 4096
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
            
            # Optimization 1: FP16 for tensor cores
            self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().eval()
            
            # Optimization 2: torch.compile for kernel fusion
            self.model = compile_model(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
            
            # Warmup compile
            test_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(test_input)
            torch.cuda.synchronize()
            
            self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            
            # Extended warmup for stable compilation
            for _ in range(50):
                with torch.no_grad():
                    _ = self.model(self.x)
            torch.cuda.synchronize()
            
            # Optimization 3: CUDA graph capture
            self.graph = None
            self.x_capture = None
            try:
                graph = torch.cuda.CUDAGraph()
                self.x_capture = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
                
                # Pre-graph warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = self.model(self.x_capture)
                torch.cuda.synchronize()
                
                # Capture graph
                with torch.cuda.graph(graph):
                    with torch.no_grad():
                        _ = self.model(self.x_capture)
                torch.cuda.synchronize()
                
                # Validate graph replay
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                self.graph = graph
            except RuntimeError as e:
                if "Cannot prepare for replay during capturing stage" in str(e):
                    warnings.warn(
                        "TorchInductor already uses CUDA graphs on this platform; skipping manual capture.",
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
                    # Use compiled model
                    _ = self.model(self.x)
                else:
                    # Use CUDA graph replay (fastest)
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
            warmup=50,  # Extra warmup for compile/graph stability
            use_subprocess=True,
        )
    
    def get_workload_metadata(self):
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization stack metrics."""
        return {
            "ch20.uses_fp16": 1.0,
            "ch20.uses_compile": 1.0,
            "ch20.uses_cuda_graph": 1.0 if self.graph is not None else 0.0,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedAllTechniquesBenchmark()
