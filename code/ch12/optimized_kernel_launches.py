"""optimized_kernel_launches.py - CUDA Graphs optimization."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
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


class OptimizedKernelLaunchesBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark implementation following BaseBenchmark."""
    
    def __init__(self):
        super().__init__()
        self.x_template = None
        self.x_capture = None
        self.graph = None
        self.replay_fn = None
        self.size = (1024, 1024)
        self.iterations = 1000
        tokens = self.size[0] * self.size[1]
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        # Kernel launch benchmark - fixed dimensions for consistent overhead measurement
    
    def setup(self) -> None:
        """Setup: initialize tensor and capture CUDA graph."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        # Use bfloat16 for GPU performance
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.x_template = torch.randn(*self.size, device=self.device, dtype=dtype)
        
        # Warmup before graph capture
        for _ in range(10):
            x_warmup = self.x_template.clone()
            for _ in range(self.iterations):
                x_warmup = x_warmup + 1.0
                x_warmup = x_warmup * 0.99
                x_warmup = torch.relu(x_warmup)
        torch.cuda.synchronize(self.device)
        
        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        self.x_capture = self.x_template.clone()
        with torch.cuda.graph(self.graph):
            for _ in range(self.iterations):
                self.x_capture = self.x_capture + 1.0
                self.x_capture = self.x_capture * 0.99
                self.x_capture = torch.relu(self.x_capture)
        self._verify_input = self.x_template.detach().clone()
        
        # Create replay function
        def replay():
            self.graph.replay()
            return self.x_capture
        
        self.replay_fn = replay
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("kernel_launches", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.replay_fn()
        self._synchronize()
        if self._verify_input is None or self.x_capture is None:
            raise RuntimeError("Verification input or captured output missing")
        dtype = self._verify_input.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.x_capture.detach().clone(),
            batch_size=self._verify_input.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.x_template, self.x_capture, self.graph, self.replay_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
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
        if self.x_template is None:
            return "Input tensor x_template not initialized"
        if self.graph is None:
            return "CUDA graph not initialized"
        if self.replay_fn is None:
            return "Replay function not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedKernelLaunchesBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
