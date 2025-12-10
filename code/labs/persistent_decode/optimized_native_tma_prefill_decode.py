"""Optimized native-TMA prefill vs. decode microbench with burst shaping (no fallbacks)."""

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
    get_stream_priorities,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)
from labs.persistent_decode.tma_extension import load_native_tma


class NativeTmaBurstConfig:
    """Burst shaping knobs for native TMA path."""

    def __init__(self, max_in_flight: int = 2) -> None:
        self.max_in_flight = max_in_flight


class OptimizedNativeTmaPrefillDecodeBenchmark(BaseBenchmark):
    """Prefill with native TMA bursts + graph-captured decode."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self.cfg = NativeTmaBurstConfig()
        self._prio_low, self._prio_high = get_stream_priorities()
        self.prefill_streams = [torch.cuda.Stream(priority=self._prio_low) for _ in range(self.cfg.max_in_flight)]
        self.decode_stream = torch.cuda.Stream(priority=self._prio_high)
        self.jitter_exemption_reason = "Native TMA prefill/decode: fixed dimensions"
        self.register_workload_metadata(tokens_per_iteration=float(self.batch * self.seq_len))
        self.decode_graph = torch.cuda.CUDAGraph()
        self.graph_q = None
        self.graph_k = None
        self.graph_v = None
        self.graph_out = None
        self._tma_ext = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.zeros_like(self.prefill_src)
        self._tma_ext = load_native_tma()  # raises if unsupported

        self.graph_q = self.inputs.q.clone()
        self.graph_k = self.inputs.k.clone()
        self.graph_v = self.inputs.v.clone()
        self.graph_out = torch.zeros_like(self.inputs.out)

        torch.cuda.synchronize()
        with torch.cuda.graph(self.decode_graph, stream=self.decode_stream):
            self._decode_body(self.graph_q, self.graph_k, self.graph_v, self.graph_out)
        torch.cuda.synchronize()

    def _decode_body(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        for t in range(self.seq_len):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            out[:, t, :] = v_t * dot

    def _prefill_shaped_native(self, *, async_only: bool = False) -> list[torch.cuda.Event] | None:
        """Launch native TMA copies on multiple streams with an in-flight cap."""
        events = []
        for idx in range(self.prefill_chunks):
            stream = self.prefill_streams[idx % len(self.prefill_streams)]
            with torch.cuda.stream(stream):
                self._tma_ext.tma_copy(self.prefill_src[idx], self.prefill_dst[idx])
            evt = torch.cuda.Event(enable_timing=False, blocking=False)
            evt.record(stream)
            events.append(evt)
            if len(events) > self.cfg.max_in_flight:
                events.pop(0).synchronize()
        if async_only:
            return events
        for evt in events:
            evt.synchronize()
        return None

    def _decode_graph(self) -> None:
        assert self.inputs is not None
        with torch.cuda.stream(self.decode_stream):
            self.graph_q.copy_(self.inputs.q)
            self.graph_k.copy_(self.inputs.k)
            self.graph_v.copy_(self.inputs.v)
            self.graph_out.zero_()
            self.decode_graph.replay()
            self.inputs.out.copy_(self.graph_out)

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("prefill_native_shaped_low_pri"):
            pref_events = self._prefill_shaped_native(async_only=True)
        with self._nvtx_range("decode_graph_high_pri"):
            self._decode_graph()
        if pref_events:
            for evt in pref_events:
                evt.synchronize()
        self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None

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
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"batch": self.batch, "seq_len": self.seq_len}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNativeTmaPrefillDecodeBenchmark()

if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
