"""Alias to reuse the optimized MoE inference benchmark implemented in ch18."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.run_vllm_decoder import VLLMMoEInferenceBenchmark  # noqa: E402


    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics for moe_inference."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 10.0),
            tpot_ms=getattr(self, '_tpot_ms', 1.0),
            total_tokens=getattr(self, '_total_tokens', 100),
            total_requests=getattr(self, '_total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

def get_benchmark():
    return VLLMMoEInferenceBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"Optimized MoE inference mean latency: {mean_ms:.3f} ms")
