"""Shared utilities for tcgen05-specific tiling benchmarks."""

from __future__ import annotations

import torch

from common.python.tcgen05_requirements import ensure_tcgen05_supported
from common.tcgen05 import load_tiling_tcgen05_module
from ch8.tiling_benchmark_base import TilingBenchmarkBase


class TilingBenchmarkBaseTCGen05(TilingBenchmarkBase):
    """Loads the SM100 tcgen05 tiling extension and uses FP16 inputs."""

    nvtx_label = "tiling_tcgen05"
    tensor_dtype = torch.float16

    def __init__(self) -> None:
        ensure_tcgen05_supported(
            loader=load_tiling_tcgen05_module,
            module_name="ch8 tiling tcgen05 kernels",
        )
        super().__init__()

    def _load_extension(self) -> None:
        self.extension = load_tiling_tcgen05_module()
