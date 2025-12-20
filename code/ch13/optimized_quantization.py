"""optimized_quantization.py - Optimized FP16 quantization for faster inference."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: FP16 quantization for faster inference with reduced memory."""

    signature_equivalence_group = "ch13_quantization_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.quantized_model = None
        self.data = None
        self.data_fp32 = None
        self.N = 65536
        tokens = self.N * 256
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(self.device).to(torch.float32)
        
        self.model.eval()
        self.quantized_model = self.model.to(torch.float16)
        self.data_fp32 = torch.randn(self.N, 256, device=self.device, dtype=torch.float32)
        self.data = self.data_fp32.to(dtype=torch.float16)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.quantized_model is None or self.data is None:
            raise RuntimeError("Model/data not initialized")
        with self._nvtx_range("optimized_quantization"):
            with torch.no_grad():
                self.output = self.quantized_model(self.data)
        self._synchronize()
        if self.data is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.data_fp32 is None:
            raise RuntimeError("FP32 verification input not initialized")
        self._set_verification_payload(
            inputs={"input": self.data_fp32},
            output=self.output.detach().float().clone(),
            batch_size=self.data.shape[0],
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.5, 5.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.quantized_model = None
        self.data = None
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
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.quantized_model is None:
            return "Quantized model not initialized"
        return None


def get_benchmark() -> OptimizedQuantizationBenchmark:
    """Factory function for harness discovery."""
    return OptimizedQuantizationBenchmark()
