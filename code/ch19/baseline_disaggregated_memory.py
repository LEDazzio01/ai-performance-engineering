"""Baseline stream-ordered single-GPU (no distributed)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class DisaggregatedBenchmark(Benchmark):
    """Baseline: Prefill+decode handled sequentially with host round-trips."""

    def __init__(self):
        self.device = resolve_device()
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.output = None
        self.hidden_dim = 256
        self.prefill_tokens = 2048
        self.decode_tokens = 128

    def setup(self) -> None:
        """Setup: Initialize single-GPU tensors."""
        torch.manual_seed(42)
        self.prefill_model = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device)
        self.decode_model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(self.device)

        self.prefill_input = torch.randn(
            self.prefill_tokens, self.hidden_dim, device=self.device, dtype=torch.float32
        )
        self.decode_input = torch.randn(
            self.decode_tokens, self.hidden_dim, device=self.device, dtype=torch.float32
        )
        self.output = torch.empty_like(self.decode_input)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Sequential prefill/decode with CPU staging."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("disaggregated_memory_baseline", enable=enable_nvtx):
            with torch.no_grad():
                prefill_out = self.prefill_model(self.prefill_input)
                newest = prefill_out[-self.decode_tokens:].contiguous()
                # Naive pipeline copies activations back to host before decode
                kv_host = newest.to("cpu", non_blocking=False)
                torch.cuda.synchronize()
                kv_gpu = kv_host.to(self.device, non_blocking=False)
                decode_seed = self.decode_input + kv_gpu
                self.output = self.decode_model(decode_seed)
            torch.cuda.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return DisaggregatedBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Disaggregated (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
