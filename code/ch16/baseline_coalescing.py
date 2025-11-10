"""baseline_coalescing.py - Baseline uncoalesced memory access in MoE context.

Demonstrates uncoalesced memory access patterns in MoE inference.
Coalescing: This baseline does not optimize memory access for coalescing.
Causes poor memory bandwidth utilization.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch6.cuda_extensions import load_coalescing_extension


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class BaselineCoalescingBenchmark(Benchmark):
    """Baseline: Uncoalesced memory access via CUDA extension kernels.

    Coalescing: This baseline does not optimize memory access for coalescing.
    Accesses memory with stride, preventing coalescing and reducing bandwidth.
    """

    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 16_000_000
        self.stride = 32  # Large stride prevents coalescing
        self._extension = None

    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        torch.manual_seed(42)
        self._extension = load_coalescing_extension()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        # Warm up once to keep compilation time out of the measurement loop.
        self._extension.uncoalesced_copy(self.output, self.input, self.stride)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Uncoalesced memory access."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self._extension is None or self.input is None or self.output is None:
            raise RuntimeError("Extension/tensors not initialized")

        with nvtx_range("baseline_coalescing_uncoalesced", enable=enable_nvtx):
            self._extension.uncoalesced_copy(self.output, self.input, self.stride)

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=4,
            setup_timeout_seconds=180,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineCoalescingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineCoalescingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: coalescing")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
