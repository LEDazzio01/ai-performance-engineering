"""Math-only FlexDecoding benchmark variant (no SDP/flash), for GB10 compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.baseline_flexdecoding import FlexDecodingHarness  # noqa: E402


class OptimizedFlexDecodingMathBenchmark(FlexDecodingHarness):
    """Math-only path: disable flash/mem-efficient SDP and FlexAttention."""

    def __init__(self) -> None:
        super().__init__(use_flex_attention=False, require_flex=False, decode_tokens=128)

    def setup(self) -> None:
        # Force SDP to math for this benchmark only.
        import torch
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        super().setup()


def get_benchmark():
    return OptimizedFlexDecodingMathBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
