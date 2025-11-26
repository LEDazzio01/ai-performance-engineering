"""Optimized AI example: fuse blocks into a single FP16 inference stack."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import enable_tf32


class OptimizedAIBenchmark(BaseBenchmark):
    """Chains the tiny blocks into one FP16 module and keeps it resident on device."""

    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.extend(
                [
                    nn.Linear(1024, 2048, bias=False),
                    nn.GELU(),
                    nn.Linear(2048, 1024, bias=False),
                ]
            )
        self.model = nn.Sequential(*layers).to(self.device).half()
        self.static_input: Optional[torch.Tensor] = None
        self.batch = 512
        self.hidden = 1024
        tokens = self.batch * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        enable_tf32()
        self.model.eval()
        self.static_input = torch.randn(self.batch, self.hidden, device=self.device, dtype=torch.float16)
        with torch.inference_mode():
            _ = self.model(self.static_input)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.static_input is not None
        with self._nvtx_range("optimized_ai"):
            with torch.inference_mode():
                _ = self.model(self.static_input)
            self._synchronize()

    def teardown(self) -> None:
        self.static_input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.static_input is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedAIBenchmark()
