"""optimized_nccl_quantization.py - GPU-side quantization with fused collectives."""

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

class OptimizedNcclQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Quantization with NCCL collective operations."""
    
    def __init__(self):
        super().__init__()
        self.tensor = None
        self.quantized = None
        self.dequantized = None
        self.stream = torch.cuda.Stream()
        self.num_chunks = 16
        self.chunk_len = 1 << 14
        self._last = 0.0
        tokens = self.num_chunks * self.chunk_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize quantized model for NCCL."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        torch.manual_seed(42)
        self.tensor = torch.randn(self.num_chunks, self.chunk_len, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Quantization operations with NCCL."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_nccl_quantization", enable=enable_nvtx):
            if self.tensor is None:
                raise RuntimeError("Tensor not initialized")
            with torch.cuda.stream(self.stream):
                max_abs = torch.amax(self.tensor.abs(), dim=1, keepdim=True).clamp(min=1e-6)
                scales = 127.0 / max_abs
                quantized = torch.clamp(torch.round(self.tensor * scales), -127, 127).to(torch.int8)
                dequant = quantized.float() / scales
                self._last = float(dequant.sum())
                self.output = dequant.clone()
            self.stream.synchronize()
        self._synchronize()
        if self.output is None or self.tensor is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.tensor},
            output=self.output,
            batch_size=self.num_chunks,
            parameter_count=0,
            precision_flags={
                "fp16": self.tensor.dtype == torch.float16,
                "bf16": self.tensor.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.tensor = None
        self.quantized = None
        self.dequantized = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedNcclQuantizationBenchmark()

if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
