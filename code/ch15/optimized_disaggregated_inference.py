"""Optimized disaggregated inference benchmark with speculative windowing.

Chapter 15: Disaggregated Inference (Optimized)

NOTE: This file uses speculative window batching to overlap decode work.
Full speculative decoding (draft model + verification) is covered in Chapter 18.
Here we focus on the disaggregation pattern; for complete speculative decoding:
- ch18/optimized_speculative_decode.py (draft model generation + verification)
- ch18/optimized_vllm_decode_graphs.py (CUDA graph optimization for decode)
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch15.baseline_disaggregated_inference import (  # noqa: E402
    _DisaggregatedInferenceBenchmark,
)


class OptimizedDisaggregatedInferenceBenchmark(_DisaggregatedInferenceBenchmark):
    """Overlap decode work by batching speculative windows."""

    def __init__(self) -> None:
        super().__init__(
            speculative_window=8,
            decode_parallelism=2,
            overlap_kv_transfer=True,
            transfer_stride=16,
        )


def get_benchmark():
    return OptimizedDisaggregatedInferenceBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
