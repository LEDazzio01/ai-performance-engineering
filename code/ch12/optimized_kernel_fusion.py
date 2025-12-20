"""optimized_kernel_fusion.py - Fused kernel using CUDA graphs (optimized)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)

# Import CUDA extension
from ch12.cuda_extensions import load_kernel_fusion_extension


class OptimizedKernelFusionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Fused kernel - single memory round trip (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.N = 16_000_000  # Larger size to be memory-bound
        self.iterations = 10
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        # Load CUDA extension (will compile on first call)
        self._extension = load_kernel_fusion_extension()
        
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)
        # Dry run so CUDA graph / kernel fusion setup cost is prepaid
        self._extension.fused_kernel(self.data, 1)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Fused kernel (single memory round trip)."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("kernel_fusion", enable=enable_nvtx):
            # Call CUDA extension with fused kernel
            self._extension.fused_kernel(self.data, self.iterations)
        self._synchronize()
        if self.data is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"data": self.data},
            output=self.data.detach().clone(),
            batch_size=self.N,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,  # Fewer iterations since kernels run internally
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take 60-90 seconds
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_graph_metrics
        return compute_graph_metrics(
            baseline_launch_overhead_us=getattr(self, '_baseline_launch_us', 10.0),
            graph_launch_overhead_us=getattr(self, '_graph_launch_us', 1.0),
            num_nodes=getattr(self, 'num_nodes', 10),
            num_iterations=getattr(self, 'num_iterations', 100),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedKernelFusionBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
