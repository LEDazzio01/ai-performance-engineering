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

from core.benchmark.verification_mixin import VerificationPayloadMixin
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


class OptimizedPipelineOverlapBenchmark(VerificationPayloadMixin, BaseBenchmark):
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
        self.output = None
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
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.stages = nn.ModuleList(
            [SimpleStage(self.hidden_dim).to(self.device).half() for _ in range(self.num_stages)]
        ).eval()
        
        self.inputs = torch.randn(
            self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16
        )
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
                # Capture output for verification
                self.output = x.detach()
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self.stages is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.output.float(),
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.stages.parameters()) if self.stages is not None else 0,
            output_tolerance=(0.1, 1.0),
        )

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

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()


def get_benchmark() -> BaseBenchmark:
    return OptimizedPipelineOverlapBenchmark()
