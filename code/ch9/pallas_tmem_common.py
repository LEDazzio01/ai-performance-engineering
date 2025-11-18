"""Shared helpers for TMEM-aware JAX Pallas benchmarks (Blackwell SM100).

This module wraps the SM100 `tcgen05_mma` path into the benchmark harness,
covering TMEM allocation, a simple K-loop, and a TMEM-aware epilogue. Baseline
and optimized wrappers live in baseline_tmem_pallas.py / optimized_tmem_pallas.py.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional, Tuple

import torch

# Optional dependency: JAX with Mosaic GPU (Pallas) support.
try:  # pragma: no cover - import guarded for availability
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax import random as jrandom
    import jax.experimental.pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    _JAX_AVAILABLE = True
    _JAX_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - defensive import guard
    jax = None
    jnp = None
    lax = None
    jrandom = None
    pl = None
    plgpu = None
    _JAX_AVAILABLE = False
    _JAX_IMPORT_ERROR = str(exc)

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclasses.dataclass(frozen=True)
class PallasTmemConfig:
    """Tuning knobs for the TMEM Pallas GEMM."""

    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    max_concurrent_steps: int = 2  # pipeline depth / double buffering
    dtype: str = "float16"
    accumulator_dtype: str = "float32"


def _pallas_env_ready() -> Tuple[bool, str]:
    """Return whether the JAX Mosaic GPU stack is usable."""
    if not _JAX_AVAILABLE:
        return False, f"SKIPPED: JAX Mosaic GPU unavailable ({_JAX_IMPORT_ERROR})"

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        return False, "SKIPPED: No GPU devices visible to JAX"

    missing_attrs = [
        name
        for name in ("TMEM", "tcgen05_mma", "copy_gmem_to_smem", "async_load_tmem")
        if not hasattr(plgpu, name)
    ]
    if missing_attrs:
        return (
            False,
            f"SKIPPED: Mosaic GPU TMEM helpers missing ({', '.join(missing_attrs)})",
        )
    return True, ""


def _build_bw_kernel(config: PallasTmemConfig):
    """Construct a TMEM accumulator kernel using tcgen05_mma."""

    tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
    barrier_slots = max(2, config.max_concurrent_steps)

    def _bw_kernel(
        a_gmem,
        b_gmem,
        out_gmem,
        acc_tmem,
        acc_smem,
        consumed_barriers,
        *,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        max_concurrent_steps: int,
    ):
        dtype = a_gmem.dtype

        mi = lax.axis_index("m")
        ni = lax.axis_index("n")

        m_slice = pl.ds(mi * tile_m, tile_m)
        n_slice = pl.ds(ni * tile_n, tile_n)

        k_iters = a_gmem.shape[1] // tile_k

        for ki in range(k_iters):
            k_slice = pl.ds(ki * tile_k, tile_k)

            # In a production kernel you'd use TMA; keep it simple here.
            a_smem = plgpu.copy_gmem_to_smem(a_gmem.at[m_slice, k_slice])
            b_smem = plgpu.copy_gmem_to_smem(b_gmem.at[k_slice, n_slice])

            arrive_slot = ki % max_concurrent_steps
            wait_slot = 1 - arrive_slot

            plgpu.tcgen05_mma(
                acc_tmem,
                a_smem,
                b_smem,
                barrier=consumed_barriers.at[arrive_slot],
                accumulate=(ki > 0),
            )
            plgpu.barrier_wait(consumed_barriers.at[wait_slot])

        # TMEM-aware epilogue: wait for the final MMA, drain TMEM, store to GMEM.
        final_barrier = 1 - (k_iters % max_concurrent_steps)
        plgpu.barrier_wait(consumed_barriers.at[final_barrier])

        acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
        plgpu.commit_smem()

        plgpu.copy_smem_to_gmem(
            acc_smem,
            out_gmem.at[m_slice, n_slice],
        )
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    return plgpu.kernel(
        functools.partial(
            _bw_kernel,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            max_concurrent_steps=config.max_concurrent_steps,
        ),
        out_shape=lambda a, b: jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), a.dtype),
        grid=lambda a, b: (
            a.shape[0] // tile_m,
            b.shape[1] // tile_n,
        ),
        grid_names=("m", "n"),
        scratch_shapes=dict(
            acc_tmem=plgpu.TMEM((tile_m, tile_n), getattr(jnp, config.accumulator_dtype)),
            acc_smem=plgpu.SMEM((tile_m, tile_n), getattr(jnp, config.dtype)),
            consumed_barriers=plgpu.Barrier(
                num_arrivals=1,
                num_barriers=barrier_slots,
                orders_tensor_core=True,
            ),
        ),
    )


class PallasTmemBenchmark(BaseBenchmark):
    """Benchmark wrapper that runs the TMEM-aware Pallas GEMM."""

    def __init__(self, config: PallasTmemConfig, *, friendly_name: str) -> None:
        super().__init__()
        available, reason = _pallas_env_ready()
        self._available = available
        self._skip_reason = reason or "SKIPPED: TMEM Pallas stack unavailable"
        self.config = config
        self.friendly_name = friendly_name
        self._kernel = None
        self._a = None
        self._b = None
        self._last_output = None

    def setup(self) -> None:
        if not self._available:
            raise RuntimeError(self._skip_reason)
        c = self.config
        m = c.tile_m * 2
        n = c.tile_n * 2
        k = c.tile_k * 2

        key = jrandom.PRNGKey(0)
        a_key, b_key = jrandom.split(key)
        dtype = getattr(jnp, c.dtype)
        self._a = jrandom.normal(a_key, (m, k), dtype=dtype)
        self._b = jrandom.normal(b_key, (k, n), dtype=dtype)

        self._kernel = _build_bw_kernel(c)
        # Trigger compilation during setup to keep benchmark iterations clean.
        _ = self._kernel(self._a, self._b).block_until_ready()
        self._synchronize()

    def _run_once(self) -> None:
        assert self._kernel is not None and self._a is not None and self._b is not None
        self._last_output = self._kernel(self._a, self._b).block_until_ready()

    def benchmark_fn(self) -> None:
        if not self._available:
            raise RuntimeError(self._skip_reason)
        with self._nvtx_range(self.friendly_name):
            self._run_once()
        self._synchronize()

    def teardown(self) -> None:
        self._kernel = None
        self._a = None
        self._b = None
        self._last_output = None
        # Clear CUDA allocator pressure from the PyTorch side.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
            timeout_seconds=120,
        )

    def validate_result(self) -> Optional[str]:
        if not self._available:
            return self._skip_reason
        if self._last_output is None:
            return "Kernel did not produce output"
        return None

