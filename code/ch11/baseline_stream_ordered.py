"""baseline_stream_ordered.py - cudaMalloc baseline for stream-ordered allocators.

This benchmark measures the cost of repeated device allocations inside a
multi-stream workload using the default (synchronizing) allocator path.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.stream_ordered import run_standard_allocator_capture
from core.profiling.nvtx_helper import canonicalize_nvtx_name


class BaselineStreamOrderedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: cudaMalloc / cudaFree inside a multi-stream loop."""

    def __init__(self) -> None:
        super().__init__()
        # Keep buffers small so allocator overhead dominates.
        self.elements = 1 << 12  # 4,096 floats (~16KB) per stream buffer
        # Many alloc/free cycles to amplify stream-ordered allocator benefit.
        self.inner_iterations = 200
        self.output: Optional[torch.Tensor] = None
        self._payload_inputs: Optional[dict] = None

        bytes_per_buffer = float(self.elements * 4)
        # Each inner-iteration: 3 streams, H2D + D2H per stream.
        self.register_workload_metadata(
            bytes_per_iteration=3.0 * float(self.inner_iterations) * bytes_per_buffer * 2.0,
            requests_per_iteration=float(self.inner_iterations),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Compile + warm the extension outside the timed region.
        _ = run_standard_allocator_capture(1024, 1)
        self._synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range("stream_ordered_baseline"):
            self.output = run_standard_allocator_capture(self.elements, self.inner_iterations)
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._payload_inputs = {
            "elements": torch.tensor([self.elements], dtype=torch.int64),
            "inner_iterations": torch.tensor([self.inner_iterations], dtype=torch.int64),
        }

    def capture_verification_payload(self) -> None:
        if self.output is None or self._payload_inputs is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs=self._payload_inputs,
            output=self.output,
            batch_size=int(self.inner_iterations),
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[canonicalize_nvtx_name("stream_ordered_baseline")],
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineStreamOrderedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
