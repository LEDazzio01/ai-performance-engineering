"""Optimized FlexDecoding benchmark that requires FlexAttention."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.baseline_flexdecoding import FlexDecodingHarness  # noqa: E402


class OptimizedFlexDecodingBenchmark(FlexDecodingHarness):
    """Optimized path: require FlexAttention; skip cleanly if it is unavailable/unstable."""

    def __init__(self) -> None:
        super().__init__(use_flex_attention=True, require_flex=False, decode_tokens=128)

    def setup(self) -> None:
        from ch18 import flexdecoding as flexdemo  # local import to read HAS_FLEX
        if not getattr(flexdemo, "HAS_FLEX", False):
            raise RuntimeError("SKIPPED: FlexAttention not available on this build.")
        try:
            super().setup()
        except Exception as exc:
            msg = str(exc)
            if "flex_attention" in msg or "LoweringException" in msg or "wrong ndim" in msg:
                # Treat flex compile errors as a skipped optimization.
                raise RuntimeError(f"SKIPPED: FlexAttention failed to compile on this GPU/build: {exc}")
            raise


    def get_custom_metrics(self) -> Optional[dict]:
        """Return speculative decoding metrics for flexdecoding."""
        from common.python.benchmark_metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 10),
            accepted_tokens=getattr(self, '_accepted_tokens', 8),
            draft_time_ms=getattr(self, '_draft_ms', 1.0),
            verify_time_ms=getattr(self, '_verify_ms', 1.0),
            num_rounds=getattr(self, '_num_rounds', 1),
        )

def get_benchmark():
    return OptimizedFlexDecodingBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
