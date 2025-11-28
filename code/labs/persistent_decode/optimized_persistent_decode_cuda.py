"""Persistent decode in CUDA via an out-of-line extension (no fallbacks)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import functools
from pathlib import Path
from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.extension_loader_template import load_cuda_extension_v2
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


@functools.lru_cache(None)
def _load_extension() -> object:
    """Compile and return the CUDA extension once per process."""
    include_dirs = [
        # Stick to the repo-pinned CUTLASS to avoid mixing cute headers from TransformerEngine.
        REPO_ROOT / "third_party" / "cutlass" / "include",
        REPO_ROOT / "common" / "headers",
    ]
    return load_cuda_extension_v2(
        name="persistent_decode_ext",
        sources=[Path(__file__).with_name("persistent_decode_ext.cu")],
        extra_cuda_cflags=[
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUTE_ARCH_TCGEN05_TMEM_ENABLED",
            "-gencode=arch=compute_100,code=sm_100",
        ] + [f"-I{p}" for p in include_dirs if p.exists()],
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
        # Explicitly skip to avoid hanging builds; this path requires a prebuilt, stable
        # persistent_decode_ext. Build offline and set PYTHONPATH/LD_LIBRARY_PATH accordingly.
        raise RuntimeError("SKIPPED: persistent_decode_ext disabled (build offline to enable)")

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: persistent_decode_ext disabled (build offline to enable)")

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

    def get_config(self) -> BenchmarkConfig:
        # NOTE: CUDA binary runs external executable, warmup=5 ensures CUDA driver is initialized
        return BenchmarkConfig(
            iterations=5,
            warmup=5,  # Required to warm CUDA driver and JIT
            use_subprocess=True,
            measurement_timeout_seconds=600,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode_cu.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode_cu.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode_cu.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeCUDABenchmark()

if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean_ms = result.timing.mean_ms if result and result.timing else 0.0
    print(f"[{bench.__class__.__name__}] mean iteration {mean_ms:.3f} ms")
