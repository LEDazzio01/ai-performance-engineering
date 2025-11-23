"""Persistent decode in CUDA via an out-of-line extension (no fallbacks)."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


@functools.lru_cache(None)
def _load_extension() -> object:
    """Compile and return the CUDA extension once per process."""
    ext_path = Path(__file__).with_name("persistent_decode_ext.cu")
    repo_root = Path(__file__).resolve().parents[2]
    include_dirs = [
        # Prefer the TransformerEngine-bundled CUTLASS (stock/upstream) to avoid local patches.
        repo_root / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
        repo_root / "common" / "headers",
        repo_root / "third_party" / "cutlass" / "include",
    ]
    return load(
        name="persistent_decode_ext",
        sources=[str(ext_path)],
        extra_cuda_cflags=[
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        + [f"-I{p}" for p in include_dirs],
        verbose=False,
    )


class OptimizedPersistentDecodeCUDABenchmark(BaseBenchmark):
    """Persistent decode using a cooperative CUDA kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.blocks = 8
        self._ext: Optional[object] = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self._ext = _load_extension()
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self._ext is None or not hasattr(self._ext, "persistent_decode"):
            raise RuntimeError("Extension or inputs not initialized")

        with self._nvtx_range("persistent_decode_cuda"):
            self._ext.persistent_decode(
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.blocks,
            )
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=4)

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeCUDABenchmark()
