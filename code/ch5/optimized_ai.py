"""Optimized AI example: fuse blocks into a single FP16 inference stack."""

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
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5 ai example")
    return torch.device("cuda")


class OptimizedAIBenchmark(Benchmark):
    """Chains the tiny blocks into one FP16 module and keeps it resident on device."""

    def __init__(self):
        self.device = resolve_device()
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

    def setup(self) -> None:
        torch.manual_seed(0)
        enable_tf32()
        self.model.eval()
        self.static_input = torch.randn(512, 1024, device=self.device, dtype=torch.float16)
        with torch.inference_mode():
            _ = self.model(self.static_input)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.static_input is not None
        with nvtx_range("optimized_ai", enable=enable_nvtx):
            with torch.inference_mode():
                _ = self.model(self.static_input)

    def teardown(self) -> None:
        self.static_input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.static_input is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedAIBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized AI latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
