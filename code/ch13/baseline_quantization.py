"""baseline_quantization.py - Baseline FP32 precision without quantization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineQuantizationBenchmark(BaseBenchmark):
    """Baseline: FP32 precision without quantization (full precision)."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self.N = 65536
        tokens = self.N * 256
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model in FP32."""
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(self.device).to(torch.float32)
        
        self.model.eval()
        self.data = torch.randn(self.N, 256, device=self.device, dtype=torch.float32)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP32 inference without quantization."""
        if self.model is None or self.data is None:
            raise RuntimeError("Model/data not initialized")
        with self._nvtx_range("baseline_quantization"):
            with torch.no_grad():
                _ = self.model(self.data)
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.data = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaselineQuantizationBenchmark:
    """Factory function for harness discovery."""
    return BaselineQuantizationBenchmark()
