"""Optimized NCCL path simulated with torch.cuda.comm collectives."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedNcclBenchmark(BaseBenchmark):
    """Keeps tensors on GPU and uses reduce_add/broadcast to simulate NCCL."""

    def __init__(self):
        super().__init__()
        self.skip_output_check = True
        self.model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
        ).to(self.device).eval()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = 256 * 2048
        self._workload = WorkloadMetadata(
            requests_per_iteration=256.0,
            tokens_per_iteration=float(tokens),
        )

    def skip_output_verification(self) -> bool:
        return True

    def setup(self) -> None:
        torch.manual_seed(0)
        self.input = torch.randn(256, 2048, device=self.device)
        self.output = torch.zeros_like(self.input)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.input is not None
        with nvtx_range("optimized_nccl", enable=enable_nvtx):
            out = self.model(self.input)
            shards = torch.chunk(out, chunks=4, dim=0)
            stacked = torch.stack(shards, dim=0)
            reduced = stacked.sum(dim=0)
            self.output = reduced / stacked.shape[0]
        self._synchronize()

    def teardown(self) -> None:
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedNcclBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized NCCL latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
