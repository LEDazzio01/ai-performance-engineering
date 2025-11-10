"""baseline_triton.py - Baseline kernel without Triton in GEMM context.

Demonstrates custom kernel without Triton optimization.
Triton: This baseline does not use Triton for kernel development.
Uses standard PyTorch operations without Triton optimizations.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from common.python.compile_utils import enable_tf32
from ch10.workload_config import WORKLOAD, is_smoke_test


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineTritonBenchmark(Benchmark):
    """Baseline: Custom kernel without Triton.
    
    Triton: This baseline does not use Triton for kernel development.
    Uses standard PyTorch operations without Triton optimizations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.residuals = None
        self.output = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.batch = self.workload.triton_batch_for_mode(self.smoke_test)
        self.micro_batches = self.workload.triton_micro_batches_for_mode(self.smoke_test)
        self.hidden_dim = self.workload.hidden_dim
        self.ffn_dim = self.workload.ffn_dim
        self._checksum = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model without Triton."""
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        enable_tf32()
        # Baseline: No Triton - standard PyTorch operations
        # Triton provides high-level DSL for writing optimized CUDA kernels
        # This baseline does not use Triton
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, self.hidden_dim, bias=True),
        ).to(self.device).half().eval()
        
        shape = (self.micro_batches, self.batch, self.hidden_dim)
        self.inputs = torch.randn(shape, device=self.device, dtype=torch.float16)
        self.residuals = torch.randn_like(self.inputs)
        self.output = torch.empty_like(self.inputs)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without Triton."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("triton", enable=enable_nvtx):
            with torch.no_grad():
                assert self.model is not None
                assert self.inputs is not None
                assert self.residuals is not None
                assert self.output is not None

                self._checksum = 0.0
                for mb in range(self.micro_batches):
                    chunk = self.inputs[mb]
                    residual = self.residuals[mb]
                    activations = self.model(chunk)
                    torch.add(
                        activations,
                        residual,
                        alpha=1.0,
                        out=self.output[mb],
                    )
                    torch.relu_(self.output[mb])
                    self._checksum += float(self.output[mb].sum().item())
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
        self.residuals = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
            enable_memory_tracking=False,
            measurement_timeout_seconds=180,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None or self.residuals is None:
            return "Tensors not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineTritonBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineTritonBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Triton")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
