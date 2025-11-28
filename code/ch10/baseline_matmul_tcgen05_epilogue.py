"""Baseline TMEM epilogue (separate matmul + bias + SiLU kernels)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineMatmulTCGen05EpilogueBenchmark(BaseBenchmark):
    """Baseline: matmul followed by separate bias addition and SiLU activation.
    
    This is the naive approach that requires:
    1. Matmul kernel (write result to global memory)
    2. Bias addition kernel (read from global, write to global)
    3. SiLU activation kernel (read from global, write to global)
    
    The optimized version fuses these into a single kernel with TMEM-resident epilogue.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda")
        self.size = 4096
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(0)
        dtype = torch.float16
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.bias = torch.randn(self.size, device=self.device, dtype=dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.A is not None and self.B is not None and self.bias is not None
        with self._nvtx_range("baseline_matmul_tcgen05_bias_silu"):
            with torch.no_grad():
                # Step 1: Matrix multiplication (uses cuBLAS)
                C = torch.matmul(self.A, self.B)
                # Step 2: Add bias (separate kernel launch)
                C = C + self.bias
                # Step 3: SiLU activation (separate kernel launch)
                C = F.silu(C)
        self._synchronize()

    def teardown(self) -> None:
        self.A = None
        self.B = None
        self.bias = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)


def get_benchmark() -> BaselineMatmulTCGen05EpilogueBenchmark:
    return BaselineMatmulTCGen05EpilogueBenchmark()

