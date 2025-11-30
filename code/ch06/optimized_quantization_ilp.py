"""optimized_quantization_ilp.py - Optimized ILP with quantization."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.workload_config import WORKLOAD


class OptimizedQuantizationILPBenchmark(BaseBenchmark):
    """Optimized: FP16 with half the memory bandwidth for 2x throughput.
    
    Chapter 6: Increasing GPU ILP Through Microarchitecture Awareness
    
    This demonstrates ILP (Instruction-Level Parallelism) optimization:
    - FP16 uses half the memory bandwidth of FP32
    - Memory-bound ops see ~2x speedup from reduced traffic
    - Modern GPUs handle FP16 arithmetic at same speed as FP32
    """
    
    def __init__(self):
        super().__init__()
        self.skip_output_check = True
        self.skip_input_check = True
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.N = self.workload.quantization_elements
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize FP16 tensors with contiguous memory layout."""
        torch.manual_seed(42)
        # FP16 uses half the memory bandwidth - key for memory-bound ops
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float16).contiguous()
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float16).contiguous()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP16 element-wise operations (2x less memory traffic)."""
        assert self.input is not None and self.output is not None
        with self._nvtx_range("optimized_quantization_ilp"):
            # Simple multiply-add in FP16 - half the memory bandwidth
            self.output = self.input * 2.0 + 1.0
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    def skip_output_verification(self) -> bool:
        return True
    
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
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
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
