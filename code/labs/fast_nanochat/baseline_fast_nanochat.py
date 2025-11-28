"""Baseline NanoChat-style decode loop."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.fast_nanochat.nanochat_common import NanoChatBenchmark, NanoChatConfig, attach_benchmark_metadata  # noqa: E402


def get_benchmark() -> NanoChatBenchmark:
    cfg = NanoChatConfig(
        batch_size=8,
        prompt_tokens=256,   # Match optimized config for fair comparison
        decode_tokens=64,    # Match optimized config for fair comparison
        hidden_size=1024,    # Match optimized config for fair comparison
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="baseline_fast_nanochat",
        iterations=12,       # Match optimized iterations
        warmup=15,           # Match optimized warmup
    )
    return attach_benchmark_metadata(NanoChatBenchmark(cfg), __file__)


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\nbaseline_fast_nanochat: {mean:.3f} ms/iter")
