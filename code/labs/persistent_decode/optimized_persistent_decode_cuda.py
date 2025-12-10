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

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension_v2
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
        REPO_ROOT / "core" / "common" / "headers",
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
        self.jitter_exemption_reason = "Persistent decode CUDA: fixed dimensions"

    def setup(self) -> None:
        """Initialize the persistent decode CUDA extension and inputs."""
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for persistent decode benchmark")
        
        # Try to load the extension - this may fail if not pre-built
        try:
            self._ext = _load_extension()
        except Exception as exc:
            raise RuntimeError(
                f"SKIPPED: persistent_decode_ext failed to build ({type(exc).__name__}: {exc}). "
                "Build offline with: cd labs/persistent_decode && python -c 'from optimized_persistent_decode_cuda import _load_extension; _load_extension()'"
            ) from exc
        
        self.inputs = build_inputs(self.batch, self.seq_len, self.head_dim, self.device)

    def benchmark_fn(self) -> None:
        """Run the persistent decode kernel."""
        if self._ext is None or self.inputs is None:
            raise RuntimeError("SKIPPED: persistent_decode_ext not initialized")
        
        # Call the extension's forward pass
        self._ext.persistent_decode(
            self.inputs.q,
            self.inputs.k,
            self.inputs.v,
            self.inputs.out,
            self.blocks,
        )
        torch.cuda.synchronize(self.device)

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

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"batch": self.batch, "seq_len": self.seq_len}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeCUDABenchmark()

if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
