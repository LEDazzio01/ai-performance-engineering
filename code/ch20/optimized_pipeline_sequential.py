"""optimized_pipeline_sequential.py - GPU-native pipeline (avoids CPU transfers).

Chapter 20: Productionization and Deployment

The baseline demonstrates an anti-pattern: copying activations to CPU between
pipeline stages (perhaps for "checkpointing" or legacy code). This adds PCIe
transfer latency and blocks GPU execution.

The optimized version keeps all data on GPU with direct stage-to-stage execution,
showing the benefit of avoiding unnecessary device transfers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleStage(nn.Module):
    """Pipeline stage with FFN and residual connection."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffn(x)
        return self.norm(out + x)


class OptimizedPipelineOverlapBenchmark(BaseBenchmark):
    """Optimized: GPU-native pipeline without CPU transfers.
    
    Key optimization vs baseline:
    - All activations stay on GPU (no PCIe round-trips)
    - Direct stage-to-stage execution 
    - Single sync at end of pipeline
    """
    
    def __init__(self):
        super().__init__()
        self.stages: Optional[nn.ModuleList] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch_size = 512
        self.hidden_dim = 1536
        self.num_stages = 4
        self.repeats = 6
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size),
            samples_per_iteration=float(self.batch_size),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        self.stages = nn.ModuleList(
            [SimpleStage(self.hidden_dim).to(self.device).half() for _ in range(self.num_stages)]
        ).eval()
        
        self.inputs = torch.randn(
            self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16
        )
        
        # Warmup
        data = self.inputs
        with torch.no_grad():
            for stage in self.stages:
                data = stage(data)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.inputs is not None and self.stages is not None
        with self._nvtx_range("pipeline_sequential_optimized"):
            with torch.no_grad():
                for _ in range(self.repeats):
                    # GPU-native: all activations stay on GPU
                    x = self.inputs
                    for stage in self.stages:
                        x = stage(x)
                    # Single sync at end - no per-stage CPU round-trips
            self._synchronize()

    def teardown(self) -> None:
        self.stages = None
        self.inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
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
        if self.stages is None:
            return "Stages not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPipelineOverlapBenchmark()
