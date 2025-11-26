"""Optimized dual-pipeline warp specialization benchmark (Chapter 11)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from common.python.hardware_capabilities import ensure_dsmem_supported  # noqa: E402


@lru_cache(maxsize=1)
def _load_optimized_extension():
    sources = [
        Path(__file__).with_name("warp_specialized_multistream_extension.cu"),
    ]
    return load(
        name="optimized_warp_specialized_two_pipelines_ext",
        sources=[str(src) for src in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class OptimizedDualPipelineBenchmark(BaseBenchmark):
    """Improved dual-pipeline warp specialization using CUDA::pipeline and additional compute warps."""

    def __init__(self) -> None:
        super().__init__()
        self.num_streams = 4
        # Skip on hardware without DSMEM/cluster support to avoid launch failures.
        ensure_dsmem_supported(description="warp-specialized cluster pipelines")
        self.ext = _load_optimized_extension()
        self.input_a: torch.Tensor | None = None
        self.input_b: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

        # Match constants from optimized_warp_specialized_two_pipelines_multistream.cu
        self.tile_elems = 1024
        self.tiles = 256
        self.baseline_total_elements = self.tiles * self.tile_elems

    def setup(self) -> None:
        torch.manual_seed(1337)
        total_elems = self.tiles * self.tile_elems
        self.input_a = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self.input_b = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input_a)
        self._synchronize()
        tokens = float(total_elems * 2)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=1.0,
        )

    def benchmark_fn(self) -> None:
        assert self.input_a is not None and self.input_b is not None
        with self._nvtx_range("optimized_dual_pipeline_multistream"):
            result = self.ext.warp_specialized_multistream_forward(
                self.input_a,
                self.input_b,
                self.num_streams,
            )
        self._synchronize()
        if self.output is not None:
            self.output.copy_(result)

    def teardown(self) -> None:
        self.input_a = None
        self.input_b = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            measurement_timeout_seconds=120,
            setup_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=getattr(self, '_sequential_ms', 10.0),
            overlapped_time_ms=getattr(self, '_overlapped_ms', 5.0),
            num_streams=getattr(self, 'num_streams', 4),
            num_operations=getattr(self, 'num_operations', 4),
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "Output tensor not initialized"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> OptimizedDualPipelineBenchmark:
    return OptimizedDualPipelineBenchmark()
