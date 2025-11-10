"""baseline_stream_ordered.py - Serial execution on default stream."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class BaselineStreamOrderedBenchmark(Benchmark):
    """Sequential work on the default stream (no overlap)."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.requests: Optional[list[torch.Tensor]] = None
        self.outputs: Optional[list[torch.Tensor]] = None
        self.batch_size = 64
        self.hidden_dim = 1024
        self.num_streams = 8

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).half().eval()

        self.requests = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(self.num_streams)
        ]
        self.outputs = [torch.empty_like(req) for req in self.requests]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None
        assert self.requests is not None and self.outputs is not None

        with nvtx_range("stream_ordered", enable=enable_nvtx):
            with torch.no_grad():
                for request, output in zip(self.requests, self.outputs):
                    output.copy_(self.model(request))
            torch.cuda.synchronize()

    def teardown(self) -> None:
        self.model = None
        self.requests = None
        self.outputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.outputs is None:
            return "Outputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineStreamOrderedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Stream Ordered: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
