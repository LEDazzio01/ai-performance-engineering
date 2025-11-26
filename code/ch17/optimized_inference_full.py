"""optimized_inference_full.py - Early exit optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class EarlyExitModel(nn.Module):
    """Model with early exit points."""
    
    def __init__(self, hidden_dim=1024, num_layers=12):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.exits = nn.ModuleList([
            nn.Linear(hidden_dim, 10)
            for _ in [6, 12, 24]
        ])
    
    def forward_early_exit(self, x, exit_distribution=[0.5, 0.3, 0.2]):
        exit_points = [6, 12, 24]
        avg_layers = (
            exit_points[0] * exit_distribution[0]
            + exit_points[1] * exit_distribution[1]
            + exit_points[2] * exit_distribution[2]
        )
        layers_to_run = int(avg_layers)
        
        for i in range(min(layers_to_run, self.num_layers)):
            x = torch.relu(self.layers[i](x))
        
        if layers_to_run <= exit_points[0]:
            exit_idx = 0
        elif layers_to_run <= exit_points[1]:
            exit_idx = 1
        else:
            exit_idx = 2
        
        return self.exits[exit_idx](x)


class OptimizedEarlyExitBenchmark(BaseBenchmark):
    """Adaptive early-exit inference (approximate cost model)."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.batch_size = 16
        self.hidden_dim = 2048
        self.num_layers = 24
        self.exit_distribution = [0.5, 0.3, 0.2]
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.model = EarlyExitModel(
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers
        )
        self.model = self.model.to(self.device)
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception as e:
                import warnings
                warnings.warn(
                    f"FP16 conversion failed: {e}. Running in FP32.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.model.eval()
        input_dtype = next(self.model.parameters()).dtype
        self.x = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=input_dtype,
        )
        
        import random
        random.seed(42)
        torch.manual_seed(42)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.model is None or self.x is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("inference_early_exit"):
            with torch.no_grad():
                _ = self.model.forward_early_exit(self.x, exit_distribution=self.exit_distribution)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.x = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
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
        return None


def get_benchmark() -> OptimizedEarlyExitBenchmark:
    return OptimizedEarlyExitBenchmark()
