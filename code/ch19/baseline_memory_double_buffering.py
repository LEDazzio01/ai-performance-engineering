"""baseline_memory_double_buffering.py - Single-stream baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class MemoryDoubleBufferingBenchmark(BaseBenchmark):
    """Baseline: single stream, single buffer (no overlap)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 1024
        self.host_batches: List[torch.Tensor] = []
        self.micro_batches = 16
        tokens = self.batch_size * self.seq_len * self.micro_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize single-GPU tensors."""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        ).to(self.device).half().eval()
        self.buffer = torch.empty(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.output = torch.empty_like(self.buffer)
        self.host_batches = [
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_dim,
                device="cpu",
                dtype=torch.float16,
            ).pin_memory()
            for _ in range(self.micro_batches)
        ]
        self.stream = torch.cuda.Stream()

    def benchmark_fn(self) -> None:
        """Benchmark: Single-GPU stream-ordered operations."""
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert (
            self.model is not None
            and self.buffer is not None
            and self.stream is not None
            and self.host_batches
        )
        with nvtx_range("baseline_memory_double_buffering", enable=enable_nvtx):
            with torch.no_grad():
                for host_batch in self.host_batches:
                    self.buffer.copy_(host_batch, non_blocking=False)
                    with torch.cuda.stream(self.stream):
                        self.output = self.model(self.buffer)
                    self.stream.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.buffer = None
        self.output = None
        self.stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return MemoryDoubleBufferingBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=MemoryDoubleBufferingBenchmark().get_config(),
    )
    benchmark = get_benchmark()
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Memory double buffering (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
