"""baseline_warp_divergence_ilp.py - Baseline ILP with warp divergence."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch6.workload_config import WORKLOAD


class BaselineWarpDivergenceILPBenchmark(BaseBenchmark):
    """Baseline: ILP limited by warp divergence."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.routing_logits: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.N = self.workload.warp_elements
        self.branch_iterations = self.workload.warp_branch_iterations
        self._checksum = 0.0
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.branch_iterations),
            tokens_per_iteration=float(self.N * self.branch_iterations),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.routing_logits = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: ILP operations with warp divergence."""
        assert self.input is not None and self.output is not None and self.routing_logits is not None
        with self._nvtx_range("baseline_warp_divergence_ilp"):
            mask_source = self.routing_logits
            result = self.input.clone()
            for iteration in range(self.branch_iterations):
                activations = torch.sigmoid(mask_source)
                mask = activations > 0.5

                positive = result[mask]
                negative = result[~mask]

                positive = torch.tanh(positive * 1.11 + 0.25)
                positive = positive * 1.003 + 0.0005 * positive * positive

                negative = torch.sin(negative * 0.77 - 0.35)
                negative = negative * 0.997 - 0.0004 * negative * negative

                result[mask] = positive
                result[~mask] = negative

                mask_source = 0.92 * mask_source + 0.08 * torch.roll(result, shifts=iteration + 1, dims=0)
                self._synchronize()

            self.output = result
            self.routing_logits = mask_source
            self._checksum = float(result.sum().item())
    
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
    """Factory function for harness discovery."""
    return BaselineWarpDivergenceILPBenchmark()
