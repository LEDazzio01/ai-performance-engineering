"""Baseline GEMM that serializes micro-batches with CPU synchronization."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import configure_tf32, restore_tf32


class BaselineGemmBenchmark(BaseBenchmark):
    """Splits a large GEMM into many small kernels with extra CPU sync."""

    def __init__(self):
        super().__init__()
        self.block = 512
        self.blocks = 8
        self.left_blocks: List[torch.Tensor] = []
        self.right_blocks: List[torch.Tensor] = []
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        tokens = self.block * self.block * self.blocks
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.blocks),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(1)
        self._tf32_state = configure_tf32(enable_matmul=False, enable_cudnn=False)
        torch.set_float32_matmul_precision("highest")
        self.left_blocks = [
            torch.randn(self.block, self.block, device=self.device, dtype=torch.float32)
            for _ in range(self.blocks)
        ]
        self.right_blocks = [
            torch.randn(self.block, self.block, device=self.device, dtype=torch.float32)
            for _ in range(self.blocks)
        ]
        self._synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        total = torch.zeros(self.block, self.block, device=self.device, dtype=torch.float32)
        with nvtx_range("baseline_gemm", enable=enable_nvtx):
            for a, b in zip(self.left_blocks, self.right_blocks):
                total += torch.matmul(a, b)
                self._synchronize()  # simulate CPU scheduling between launches
        return total

    def teardown(self) -> None:
        self.left_blocks = []
        self.right_blocks = []
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self):
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if not self.left_blocks or not self.right_blocks:
            return "Blocks not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineGemmBenchmark()
