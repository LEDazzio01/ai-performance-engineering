"""Baseline native-TMA prefill vs. decode microbench (no fallbacks)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)
from labs.persistent_decode.tma_extension import load_native_tma


class BaselineNativeTmaPrefillDecodeBenchmark(BaseBenchmark):
    """Sequential native-TMA copy/compute prefill + host decode."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.output: Optional[torch.Tensor] = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self._tma_ext = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.zeros_like(self.prefill_src)
        self._tma_ext = load_native_tma()  # raises if unsupported
        self._synchronize()

    def _prefill_native(self) -> None:
        for idx in range(self.prefill_chunks):
            self._tma_ext.tma_copy(self.prefill_src[idx], self.prefill_dst[idx])

    def _decode_host_loop(self) -> None:
        assert self.inputs is not None
        for t in range(self.seq_len):
            q_t = self.inputs.q[:, t, :]
            k_t = self.inputs.k[:, t, :]
            v_t = self.inputs.v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            self.inputs.out[:, t, :] = v_t * dot

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("prefill_native_baseline"):
            self._prefill_native()
        with self._nvtx_range("decode_baseline"):
            self._decode_host_loop()
        self._synchronize()
        if self.inputs is not None:
            self.output = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])].detach().float().clone()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            use_subprocess=False,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "native_tma_prefill_d.batch_size": float(getattr(self, 'batch_size', 0)),
            "native_tma_prefill_d.seq_len": float(getattr(self, 'seq_len', 0)),
            "native_tma_prefill_d.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {
            "batch": self.batch,
            "seq_len": self.seq_len,
            "head_dim": self.head_dim,
            "prefill_chunks": self.prefill_chunks,
            "prefill_chunk_elems": self.prefill_chunk_elems,
            "shapes": {
                "prefill_src": (self.prefill_chunks, self.prefill_chunk_elems),
                "prefill_dst": (self.prefill_chunks, self.prefill_chunk_elems),
                "q": (self.batch, self.seq_len, self.head_dim),
                "k": (self.batch, self.seq_len, self.head_dim),
                "v": (self.batch, self.seq_len, self.head_dim),
                "out": (self.batch, self.seq_len, self.head_dim),
            },
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineNativeTmaPrefillDecodeBenchmark()

if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
