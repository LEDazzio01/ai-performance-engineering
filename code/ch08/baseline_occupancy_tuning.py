"""Benchmark wrapper for the occupancy_tuning CUDA binary."""

from __future__ import annotations
from typing import Optional

import os
from pathlib import Path

repo_root = Path(__file__).parent.parent
import sys

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.cuda_binary_benchmark import (  # noqa: E402
    ARCH_SUFFIX,
    CudaBinaryBenchmark,
    detect_supported_arch,
)


class OccupancyBinaryBenchmark(CudaBinaryBenchmark):
    """Wraps occupancy_tuning.cu so it runs under the standard harness."""

    def __init__(
        self,
        *,
        build_env: dict[str, str] | None = None,
        friendly_name: str,
        run_args: list[str] | None = None,
    ):
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="occupancy_tuning",
            friendly_name=friendly_name,
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            # Baseline: small block, heavy shared memory to depress occupancy.
            run_args=run_args
            or [
                "--block-size",
                "32",
                "--smem-bytes",
                "45000",
                "--unroll",
                "1",
                "--inner-iters",
                "1",
                "--reps",
                "60",
            ],
            time_regex=r"avg_kernel_ms=([0-9]+\.?[0-9]*)",  # Parse kernel time from binary output.
        )
        self.build_env = build_env or {}

    def _build_binary(self) -> None:
        """Compile the executable with optional env overrides (e.g., MAXRREGCOUNT)."""
        self.arch = detect_supported_arch()
        suffix = ARCH_SUFFIX[self.arch]
        target = f"{self.binary_name}{suffix}"
        env = os.environ.copy()
        env.update(self.build_env)
        build_cmd = ["make", f"ARCH={self.arch}", target]

        completed = self._run_make(build_cmd, env)
        path = self.chapter_dir / target
        if not path.exists():
            raise FileNotFoundError(f"Built binary not found at {path}")
        self.exec_path = path

    def _run_make(self, build_cmd, env):
        import subprocess

        try:
            completed = subprocess.run(
                build_cmd,
                cwd=self.chapter_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Build timeout: {' '.join(build_cmd)} exceeded 60 seconds")

        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to build {build_cmd[-1]} (arch={self.arch}).\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed


class BaselineOccupancyTuningBenchmark(OccupancyBinaryBenchmark):
    def __init__(self) -> None:
        super().__init__(friendly_name="Occupancy Tuning (baseline)")


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for occupancy_tuning."""
        from core.benchmark.metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="occupancy_tuning",
        )

def get_benchmark() -> BaselineOccupancyTuningBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineOccupancyTuningBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"\nOccupancy Tuning (baseline): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
