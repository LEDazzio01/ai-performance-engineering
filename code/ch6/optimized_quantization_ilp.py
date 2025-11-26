"""optimized_quantization_ilp.py - Optimized ILP with quantization."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch6.workload_config import WORKLOAD


class OptimizedQuantizationILPBenchmark(BaseBenchmark):
    """Optimized: BF16 with fused multiply-add for higher throughput."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.N = self.workload.quantization_elements
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
        # Scalar tensor for fused multiply-add
        self._scale: Optional[torch.Tensor] = None
        self._bias: Optional[torch.Tensor] = None
    
    def setup(self) -> None:
        """Setup: Initialize BF16 tensors with contiguous memory layout."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        torch.manual_seed(42)
        # BF16 is better optimized on Blackwell than FP16 for compute
        self.input = torch.randn(self.N, device=self.device, dtype=torch.bfloat16).contiguous()
        self.output = torch.empty(self.N, device=self.device, dtype=torch.bfloat16).contiguous()
        # Pre-allocate scalar tensors to avoid allocation overhead
        self._scale = torch.tensor(2.0, device=self.device, dtype=torch.bfloat16)
        self._bias = torch.tensor(1.0, device=self.device, dtype=torch.bfloat16)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: BF16 fused multiply-add operations."""
        assert self.input is not None and self.output is not None
        with self._nvtx_range("optimized_quantization_ilp"):
            # Use addcmul for fused multiply-add: output = bias + input * scale
            torch.addcmul(self._bias, self.input, self._scale, out=self.output)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedQuantizationILPBenchmark()
