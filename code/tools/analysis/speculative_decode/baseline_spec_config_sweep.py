"""Baseline: run vLLM-style MoE decoder with default speculator config."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ch18.run_vllm_decoder import DEFAULT_SPEC_CONFIG, GraphMode, VLLMMoEInferenceBenchmark  # noqa: E402


def get_benchmark() -> VLLMMoEInferenceBenchmark:
    bench = VLLMMoEInferenceBenchmark()
    bench.spec_config_path = DEFAULT_SPEC_CONFIG
    bench.graph_mode = GraphMode.EAGER
    bench.enable_graphs = False
    return bench


def main() -> None:
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    harness.benchmark(bench)


if __name__ == "__main__":
    main()
