"""Optimized: Dual streams for overlapped copy and compute."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata  # noqa: E402


def get_benchmark() -> DecodeBenchmark:
    """Dual-stream decode with overlapped copy and compute.
    
    Optimization: Uses separate CUDA streams for memory operations vs compute,
    allowing overlap and hiding transfer latency with computation.
    """
    cfg = DecodeConfig(
        batch_size=8,
        prompt_tokens=256,
        decode_tokens=64,
        hidden_size=1024,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        label="optimized_decode_streams",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\noptimized_decode_streams: {mean:.3f} ms/iter")




