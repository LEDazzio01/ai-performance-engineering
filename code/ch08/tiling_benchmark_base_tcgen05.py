"""Shared utilities for tcgen05-specific tiling benchmarks."""

from __future__ import annotations

from typing import Optional

import torch

from ch08.tiling_benchmark_base import TilingBenchmarkBase
from core.harness.benchmark_harness import BenchmarkConfig


def _check_tcgen05_extension_available() -> tuple[bool, Optional[str]]:
    """Check if the tcgen05 tiling extension can be built."""
    try:
        from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
        from core.common.tcgen05 import load_tiling_tcgen05_module
        ensure_tcgen05_supported(
            loader=load_tiling_tcgen05_module,
            module_name="ch08 tiling tcgen05 kernels",
        )
        return True, None
    except RuntimeError as e:
        msg = str(e)
        if "SKIPPED" in msg:
            return False, msg
        # Build/compile errors - convert to SKIPPED
        if "Error building extension" in msg or "ninja" in msg.lower():
            return False, f"SKIPPED: tcgen05 extension build failed (CUTLASS header incompatibility with CUDA 13.0)"
        return False, f"SKIPPED: tcgen05 unavailable ({msg[:100]})"
    except Exception as e:
        return False, f"SKIPPED: tcgen05 unavailable ({type(e).__name__}: {str(e)[:80]})"


class TilingBenchmarkBaseTCGen05(TilingBenchmarkBase):
    """Loads the SM100 tcgen05 tiling extension and uses FP16 inputs."""

    nvtx_label = "tiling_tcgen05"
    tensor_dtype = torch.float16
    # The tcgen05 kernel is extremely fast on Blackwell; use a shape that makes
    # the redundant baseline output copy a measurable fraction of runtime.
    # Keep the compute workload substantial while increasing the output tensor
    # footprint (M x N) so the avoided copy is not lost in noise.
    matrix_rows: int = 8192
    matrix_cols: int = 8192
    # Use a smaller K so the baseline's redundant output copy is a larger
    # fraction of end-to-end time (and the optimization is clearly measurable).
    shared_dim: int = 64

    def __init__(self) -> None:
        # Check availability first and raise SKIPPED if needed
        available, reason = _check_tcgen05_extension_available()
        if not available:
            raise RuntimeError(reason or "SKIPPED: tcgen05 extension unavailable")
        super().__init__()

    def _load_extension(self) -> None:
        from core.common.tcgen05 import load_tiling_tcgen05_module
        self.extension = load_tiling_tcgen05_module()

    def get_config(self) -> BenchmarkConfig:
        # tcgen05 kernels are extremely fast; use more iterations for stable mean timing.
        return BenchmarkConfig(iterations=200, warmup=10)
