"""Persistent decode with CUDA Graphs: prefill + decode captured separately."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enum import Enum

import torch
import triton
import triton.language as tl

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.optimized_persistent_decode_triton import persistent_decode_kernel
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    get_decode_options,
    get_decode_profile,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class GraphMode(Enum):
    FULL = "full"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"

    @classmethod
    def from_str(cls, raw: str | None) -> "GraphMode":
        normalized = (raw or cls.FULL_AND_PIECEWISE.value).strip().lower().replace("-", "_")
        for mode in cls:
            if normalized == mode.value:
                return mode
        return cls.FULL_AND_PIECEWISE


class OptimizedPersistentDecodeGraphsBenchmark(BaseBenchmark):
    """Capture prefill and persistent decode in CUDA Graphs."""

    def __init__(self, *, graph_mode: GraphMode | None = None, max_capture_seq: int | None = None) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.options = get_decode_options()
        self.profile = get_decode_profile()
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.block_k = self.profile.block_k
        self.num_programs = self.profile.num_programs
        self.prefill_graph: torch.cuda.CUDAGraph | None = None
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self.full_graph: torch.cuda.CUDAGraph | None = None
        self.prefill_out: torch.Tensor | None = None
        self.graph_mode = graph_mode or GraphMode.FULL_AND_PIECEWISE
        self.max_capture_seq = max_capture_seq or self.seq_len
        self._history: dict[str, list[float]] = {}
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        self.inputs = build_inputs(self.device)
        self.prefill_out = torch.empty((self.batch, self.seq_len), device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
        self._capture_graphs()

    def _capture_graphs(self) -> None:
        self._capture_piecewise_graphs()
        self._capture_full_graph()

    def _capture_piecewise_graphs(self) -> None:
        # Capture prefill (toy: dot across head_dim per token)
        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph):
            qk = (self.inputs.q * self.inputs.k).sum(dim=-1)
            self.prefill_out.copy_(qk)

        # Capture persistent decode kernel
        self.decode_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        num_items = min(self.batch, self.num_programs)
        grid = (max(1, num_items),)
        BLOCK_K = self.block_k
        with torch.cuda.graph(self.decode_graph):
            persistent_decode_kernel[grid](
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.inputs.work_seq_ids,
                self.inputs.work_steps,
                num_items,
                head_dim=self.head_dim,
                max_steps=self.seq_len,
                BLOCK_K=BLOCK_K,
                num_warps=2,
                num_stages=1,
            )
        torch.cuda.synchronize()

    def _capture_full_graph(self) -> None:
        if self.graph_mode == GraphMode.PIECEWISE:
            return
        self.full_graph = torch.cuda.CUDAGraph()
        num_items = min(self.batch, self.num_programs)
        grid = (max(1, num_items),)
        BLOCK_K = self.block_k
        with torch.cuda.graph(self.full_graph):
            self.inputs.out.zero_()
            qk = (self.inputs.q * self.inputs.k).sum(dim=-1)
            self.prefill_out.copy_(qk)
            persistent_decode_kernel[grid](
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.inputs.work_seq_ids,
                self.inputs.work_steps,
                num_items,
                head_dim=self.head_dim,
                max_steps=self.seq_len,
                BLOCK_K=BLOCK_K,
                num_warps=2,
                num_stages=1,
            )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.prefill_out is None:
            raise RuntimeError("Inputs not initialized")

        use_full = (
            self.graph_mode == GraphMode.FULL
            or (self.graph_mode == GraphMode.FULL_AND_PIECEWISE and self.seq_len <= self.max_capture_seq)
        )
        if use_full and self.full_graph is not None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with self._nvtx_range("full_graph"):
                start.record()
                self.full_graph.replay()
                end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            self._history.setdefault("ttft_ms", []).append(total_ms)
            self._history.setdefault("decode_ms", []).append(total_ms)
            self._history.setdefault("per_token_ms", []).append(total_ms / max(1, self.seq_len))
            self._history.setdefault("graph_path", []).append("full_graph")
            return

        if self.prefill_graph is None or self.decode_graph is None:
            raise RuntimeError("Piecewise graphs not initialized")

        with self._nvtx_range(
            "piecewise_graph" if self.graph_mode != GraphMode.FULL_AND_PIECEWISE else "graph_fallback_piecewise"
        ):
            start_prefill = torch.cuda.Event(enable_timing=True)
            end_prefill = torch.cuda.Event(enable_timing=True)
            start_decode = torch.cuda.Event(enable_timing=True)
            end_decode = torch.cuda.Event(enable_timing=True)
            start_prefill.record()
            self.prefill_graph.replay()
            end_prefill.record()
            start_decode.record()
            self.decode_graph.replay()
            end_decode.record()
            torch.cuda.synchronize()
            ttft_ms = start_prefill.elapsed_time(end_prefill)
            decode_ms = start_decode.elapsed_time(end_decode)
            self._history.setdefault("ttft_ms", []).append(ttft_ms)
            self._history.setdefault("decode_ms", []).append(decode_ms)
            self._history.setdefault("per_token_ms", []).append(decode_ms / max(1, self.seq_len))
            self._history.setdefault("graph_path", []).append("piecewise_graph")
        self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.prefill_graph = None
        self.decode_graph = None
        self.full_graph = None
        self.prefill_out = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=3,
            warmup=5,
            enable_profiling=False,
            enable_ncu=False,
            enable_nsys=False,
            use_subprocess=False,
            measurement_timeout_seconds=90,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode_gr.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode_gr.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode_gr.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeGraphsBenchmark()

if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean_ms = result.timing.mean_ms if result and result.timing else 0.0
    print(f"[{bench.__class__.__name__}] mean iteration {mean_ms:.3f} ms")
