"""Python harness wrapper for the CUDA optimized_matmul_tiled binary."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedMatmulCudaBenchmark(CudaBinaryBenchmark):
    """Wraps the tiled resident matmul CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n = 1024
        bytes_a = n * n * 4
        bytes_b = n * n * 4
        bytes_c = n * n * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_matmul_tiled",
            friendly_name="Ch7 Optimized Matmul CUDA",
            iterations=3,
            warmup=5,
            timeout_seconds=180,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "N": n,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(bytes_a + bytes_b + bytes_c),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> OptimizedMatmulCudaBenchmark:
    return OptimizedMatmulCudaBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
