"""Baseline roofline quantization – scalar CPU path plus redundant memory traffic."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch14")
    return torch.device("cuda")


class BaselineRooflineQuantizationBenchmark(Benchmark):
    """Baseline: emulate quantization with scalar CPU loops + extra memory traffic."""

    def __init__(self):
        self.device = resolve_device()
        self.tensor = None
        self.chunk_len = 1 << 13
        self.num_chunks = 32
        self._last = 0.0

    def setup(self) -> None:
        """Setup: Initialize synthetic activations."""
        torch.manual_seed(42)
        self.tensor = torch.randn(self.num_chunks, self.chunk_len, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Scalar quantization with redundant host/device copies."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_roofline_quantization", enable=enable_nvtx):
            if self.tensor is None:
                raise RuntimeError("Tensor not initialized")
            total = 0.0
            for idx in range(self.num_chunks):
                chunk = self.tensor[idx].detach().cpu().numpy()
                max_abs = max(abs(chunk).max(), 1e-6)
                scale = 127.0 / max_abs
                q = (chunk * scale).round().clip(-127, 127).astype("int8")
                dq = (q.astype("float32") / scale).astype("float32")
                total += float(dq.sum())
                self.tensor[idx].copy_(torch.from_numpy(dq).to(self.device))
            self._last = total
            torch.cuda.synchronize(self.device)


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.tensor = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineRooflineQuantizationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Roofline Quantization (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
