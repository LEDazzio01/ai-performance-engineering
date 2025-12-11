"""Baseline UMA memory reporting using cudaMemGetInfo only."""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Optional

import torch
import typer

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from ch02.uma_memory_utils import format_bytes, is_integrated_gpu, read_meminfo


class BaselineUmaMemoryReportingBenchmark(BaseBenchmark):
    """Relies solely on cudaMemGetInfo with no UMA adjustment."""

    def __init__(self):
        super().__init__()
        self.cuda_free_bytes = 0
        self.cuda_total_bytes = 0
        self.host_available_bytes: Optional[int] = None
        self.swap_free_bytes: Optional[int] = None
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        torch.cuda.empty_cache()
        self._sample()

    def _sample(self) -> None:
        free, total = torch.cuda.mem_get_info()
        self.cuda_free_bytes = free
        self.cuda_total_bytes = total
        snapshot = read_meminfo()
        if snapshot:
            self.host_available_bytes = snapshot.effective_available_kb() * 1024
            self.swap_free_bytes = snapshot.swap_free_kb * 1024
        else:
            self.host_available_bytes = None
            self.swap_free_bytes = None

    def benchmark_fn(self) -> None:
        self._sample()
        # Surface a perturbable tensor using the sampled memory metrics
        values = [
            float(self.cuda_free_bytes),
            float(self.cuda_total_bytes),
            float(self.host_available_bytes or 0),
            float(self.swap_free_bytes or 0),
        ]
        summary_tensor = torch.tensor([values], dtype=torch.float32)
        if self.metrics is None or tuple(self.metrics.shape) != tuple(summary_tensor.shape):
            self.metrics = torch.randn_like(summary_tensor)
        self.output = (summary_tensor + self.metrics).detach()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            enable_memory_tracking=False,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        metrics: Dict[str, float] = {
            "cuda_free_gb": self.cuda_free_bytes / (1024**3),
            "cuda_total_gb": self.cuda_total_bytes / (1024**3),
        }
        if self.host_available_bytes is not None:
            metrics["host_memavailable_gb"] = self.host_available_bytes / (1024**3)
        if self.swap_free_bytes is not None:
            metrics["swap_free_gb"] = self.swap_free_bytes / (1024**3)
        return metrics

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        shape = tuple(self.metrics.shape) if self.metrics is not None else (1, 4)
        return {"type": "uma_memory_baseline", "shapes": {"metrics": shape}}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)

    def teardown(self) -> None:
        self.metrics = None
        self.output = None
        super().teardown()


def summarize() -> None:
    bench = BaselineUmaMemoryReportingBenchmark()
    bench.setup()
    bench.benchmark_fn()
    integrated = is_integrated_gpu()
    print("\n=== Baseline CUDA memory report ===")
    print(f"Integrated GPU detected: {integrated}")
    print(f"cudaMemGetInfo free:  {format_bytes(bench.cuda_free_bytes)}")
    print(f"cudaMemGetInfo total: {format_bytes(bench.cuda_total_bytes)}")
    if bench.host_available_bytes is not None:
        print(f"Host MemAvailable:   {format_bytes(bench.host_available_bytes)}")
    if bench.swap_free_bytes is not None:
        print(f"SwapFree:            {format_bytes(bench.swap_free_bytes)}")


def get_benchmark() -> BaseBenchmark:
    return BaselineUmaMemoryReportingBenchmark()


app = typer.Typer(help="Baseline UMA memory reporting (cudaMemGetInfo only).")


@app.command("report")
def report() -> None:
    """Print baseline UMA memory snapshot."""
    summarize()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
