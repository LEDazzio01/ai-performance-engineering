"""Baseline AI optimization example: repeated CPU-bound orchestration."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class TinyBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class BaselineAIBenchmark(BaseBenchmark):
    """Runs several tiny blocks sequentially with CPU sync between them."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(1024).to(self.device) for _ in range(4))
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 512
        self.hidden = 1024
        tokens = self.batch * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.inputs = torch.randn(self.batch, self.hidden, device=self.device, dtype=torch.float32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.inputs is not None
        with self._nvtx_range("baseline_ai"):
            out = self.inputs
            for block in self.blocks:
                out = block(out)
                self._synchronize()
        return out

    def teardown(self) -> None:
        self.inputs = None
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
        if self.inputs is None:
            return "Inputs missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineAIBenchmark()
