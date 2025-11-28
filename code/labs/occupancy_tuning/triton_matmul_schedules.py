"""Harness-friendly Triton matmul benchmark with Proton defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.occupancy_tuning import triton_matmul


@dataclass(frozen=True)
class MatmulSchedule:
    name: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    notes: str


BASELINE_SCHEDULE = MatmulSchedule(
    name="bm128_bn128_bk64",
    block_m=128,
    block_n=128,
    block_k=64,
    num_warps=4,
    notes="Reference tile that spikes register count and often falls off predicted occupancy.",
)

OPTIMIZED_SCHEDULE = MatmulSchedule(
    name="bm64_bn256_bk32",
    block_m=64,
    block_n=256,
    block_k=32,
    num_warps=4,
    notes="Tile that generally trims registers/thread and achieves higher active warps.",
)

EXTRA_SCHEDULE = MatmulSchedule(
    name="bm128_bn256_bk64",
    block_m=128,
    block_n=256,
    block_k=64,
    num_warps=4,
    notes="Wide-N tile meant to highlight when shared memory pressure caps theoretical occupancy.",
)

WARP_HEAVY_SCHEDULE = MatmulSchedule(
    name="bm128_bn128_bk32_nw8",
    block_m=128,
    block_n=128,
    block_k=32,
    num_warps=8,
    notes="Doubles num_warps to show when higher warp count exacerbates register pressure but boosts latency hiding when resources allow.",
)

LATENCY_FRIENDLY_SCHEDULE = MatmulSchedule(
    name="bm64_bn64_bk32_nw2",
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=2,
    notes="Small tile that minimizes per-block resources so theoretical occupancy approaches 100% (useful for verifying Proton vs Nsight agreement).",
)

SCHEDULES = [
    BASELINE_SCHEDULE,
    OPTIMIZED_SCHEDULE,
    EXTRA_SCHEDULE,
    WARP_HEAVY_SCHEDULE,
    LATENCY_FRIENDLY_SCHEDULE,
]


def _ensure_inductor_env() -> None:
    os.environ.setdefault("TORCHINDUCTOR_PROTON_LOGDIR", ".proton")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "1")


class TritonMatmulProtonBenchmark(BaseBenchmark):
    """Wrap a Triton matmul run so the harness can drive Proton automatically."""

    def __init__(
        self,
        schedule: MatmulSchedule,
        *,
        size: int = 4096,
        iterations: int = 2,
        warmup: int = 10,
        dtype: torch.dtype = torch.float16,
        use_compile: bool = True,
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self._size_m = size
        self._size_n = size
        self._size_k = size
        self._dtype = dtype
        self._use_compile = use_compile
        self._runner: Optional[Callable[[], torch.Tensor]] = None
        self._output: Optional[torch.Tensor] = None
        self._reference: Optional[torch.Tensor] = None
        self._a: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None
        self._scratch: Optional[torch.Tensor] = None
        self._config = BenchmarkConfig(
            iterations=iterations,
            warmup=warmup,
            enable_nvtx=True,
            enable_profiling=True,
            enable_nsys=False,
            enable_ncu=False,
            enable_proton=True,
            profile_type="minimal",
            target_label=f"labs/occupancy_tuning:{schedule.name}",
            use_subprocess=True,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: Triton Proton lab requires a CUDA device.")

        device = torch.device("cuda")
        _ensure_inductor_env()

        torch.manual_seed(0)
        self._a = torch.randn((self._size_m, self._size_k), dtype=self._dtype, device=device)
        self._b = torch.randn((self._size_k, self._size_n), dtype=self._dtype, device=device)
        self._scratch = torch.empty((self._size_m, self._size_n), dtype=self._dtype, device=device)
        with torch.no_grad():
            self._reference = torch.matmul(self._a, self._b)

        def _run_once() -> torch.Tensor:
            assert self._a is not None and self._b is not None and self._scratch is not None
            return triton_matmul.run_one(
                M=self._size_m,
                N=self._size_n,
                K=self._size_k,
                bm=self.schedule.block_m,
                bn=self.schedule.block_n,
                bk=self.schedule.block_k,
                nw=self.schedule.num_warps,
                dtype=self._dtype,
                device=device,
                a=self._a,
                b=self._b,
                c=self._scratch,
            )

        runner: Callable[[], torch.Tensor]
        if self._use_compile and hasattr(torch, "compile"):
            try:
                runner = torch.compile(_run_once, fullgraph=True)  # type: ignore[arg-type]
            except Exception:
                runner = _run_once
        else:
            runner = _run_once

        self._runner = runner

    def benchmark_fn(self) -> None:
        assert self._runner is not None
        with self._nvtx_range(self.schedule.name):
            self._output = self._runner()
        torch.cuda.synchronize()

    def validate_result(self) -> Optional[str]:
        if self._reference is None or self._output is None:
            return None
        diff = (self._output - self._reference).abs().max().item()
        if torch.isnan(self._output).any():
            return "NaNs detected in Triton output"
        if diff > 2.0:
            return f"Max diff {diff:.4f} exceeds tolerance"
        return None

    def teardown(self) -> None:
        self._runner = None
        self._output = None
        self._reference = None
        self._a = None
        self._b = None
        self._scratch = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return self._config

    def get_custom_metrics(self) -> Optional[dict]:
        """Return Triton matmul schedule and roofline metrics."""
        M, N, K = self._size_m, self._size_n, self._size_k
        flops = 2.0 * M * N * K  # MAD operations
        bytes_transferred = (M * K + K * N + M * N) * 2.0  # fp16
        arithmetic_intensity = flops / bytes_transferred if bytes_transferred > 0 else 0.0
        return {
            f"triton.{self.schedule.name}.block_m": float(self.schedule.block_m),
            f"triton.{self.schedule.name}.block_n": float(self.schedule.block_n),
            f"triton.{self.schedule.name}.block_k": float(self.schedule.block_k),
            f"triton.{self.schedule.name}.num_warps": float(self.schedule.num_warps),
            f"triton.{self.schedule.name}.matrix_size": float(M),
            f"triton.{self.schedule.name}.flops": flops,
            f"triton.{self.schedule.name}.arithmetic_intensity": arithmetic_intensity,
        }


__all__ = [
    "MatmulSchedule",
    "TritonMatmulProtonBenchmark",
    "BASELINE_SCHEDULE",
    "OPTIMIZED_SCHEDULE",
    "EXTRA_SCHEDULE",
    "WARP_HEAVY_SCHEDULE",
    "LATENCY_FRIENDLY_SCHEDULE",
    "SCHEDULES",
]
