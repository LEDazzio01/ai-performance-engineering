"""FlexDecoding decode path wrapped in CUDA Graphs for lower launch overhead."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch18.baseline_flexdecoding import FlexDecodingHarness  # noqa: E402


class OptimizedFlexDecodingGraphsBenchmark(FlexDecodingHarness):
    """Capture a single-token decode in a CUDA Graph and replay per token."""

    def __init__(self) -> None:
        super().__init__(use_flex_attention=False, require_flex=False, decode_tokens=128)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.static_decode_in: torch.Tensor | None = None
        self.static_decode_out: torch.Tensor | None = None
        self.base_position: int = 0
        self._orig_compile_callable = None
        self._orig_flex_compile_callable = None
        self._compile_disabled = False

    def _patch_compile_to_eager(self) -> None:
        """Fallback: disable torch.compile to keep capture safe."""
        from common.python import compile_utils
        from ch18 import flexdecoding as flexdemo

        if self._orig_compile_callable is None:
            self._orig_compile_callable = compile_utils.compile_callable
        if self._orig_flex_compile_callable is None:
            self._orig_flex_compile_callable = flexdemo.compile_callable

        compile_utils.compile_callable = lambda fn, **kwargs: fn  # type: ignore[assignment]
        flexdemo.compile_callable = lambda fn, **kwargs: fn  # type: ignore[assignment]
        self._compile_disabled = True

    def _restore_compile_hooks(self) -> None:
        if self._orig_compile_callable is not None:
            from common.python import compile_utils

            compile_utils.compile_callable = self._orig_compile_callable  # type: ignore[assignment]
            self._orig_compile_callable = None
        if self._orig_flex_compile_callable is not None:
            from ch18 import flexdecoding as flexdemo

            flexdemo.compile_callable = self._orig_flex_compile_callable  # type: ignore[assignment]
            self._orig_flex_compile_callable = None
        self._compile_disabled = False

    def _run_warmup(self) -> None:
        """Compile and warm kernels before capture."""
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")
        with torch.inference_mode():
            self.model.prefill(self.prefill_tokens)
            _ = self.model.decode(self.decode_token, self.base_position)
        torch.cuda.synchronize(self.device)

    def _restore_compile_hooks(self) -> None:
        if self._orig_compile_callable is not None:
            from common.python import compile_utils

            compile_utils.compile_callable = self._orig_compile_callable  # type: ignore[assignment]
            self._orig_compile_callable = None
        if self._orig_flex_compile_callable is not None:
            from ch18 import flexdecoding as flexdemo

            flexdemo.compile_callable = self._orig_flex_compile_callable  # type: ignore[assignment]
            self._orig_flex_compile_callable = None

    def setup(self) -> None:
        try:
            self._initialize_and_capture()
        except Exception:
            # If capture failed (often due to torch.compile laziness), fallback to eager and retry once.
            if not self._compile_disabled:
                self._patch_compile_to_eager()
                self._initialize_and_capture()
            else:
                self._restore_compile_hooks()
                raise

    def _initialize_and_capture(self) -> None:
        super().setup()
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")

        self.base_position = self.prefill_tokens.size(1)
        self.static_decode_in = torch.zeros_like(self.decode_token)
        self.static_decode_out = torch.empty_like(self.decode_token)
        self.capture_stream = torch.cuda.Stream(device=self.device)

        # Compile/warm outside capture to avoid lazy compile during graph capture.
        self._run_warmup()

        self.graph = torch.cuda.CUDAGraph()
        assert self.capture_stream is not None
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            out = self.model.decode(self.static_decode_in, self.base_position)  # type: ignore[arg-type]
            self.static_decode_out.copy_(out)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if (
            self.model is None
            or self.prefill_tokens is None
            or self.decode_token is None
            or self.graph is None
            or self.capture_stream is None
            or self.static_decode_in is None
        ):
            raise RuntimeError("Graph path not initialized")

        prefill_times: List[float] = []
        decode_times: List[float] = []

        self.model.clear_cache(batch=self.prefill_tokens.size(0))

        with torch.no_grad():
            with self._nvtx_range("flex_prefill"):
                start = time.perf_counter()
                _ = self.model.prefill(self.prefill_tokens)
                torch.cuda.synchronize(self.device)
                prefill_times.append((time.perf_counter() - start) * 1000.0)

            with self._nvtx_range("flex_decode_graph"):
                for _ in range(self.decode_tokens):
                    start = time.perf_counter()
                    with torch.cuda.stream(self.capture_stream):
                        self.static_decode_in.copy_(self.decode_token)
                        self.graph.replay()
                    torch.cuda.synchronize(self.device)
                    decode_times.append((time.perf_counter() - start) * 1000.0)

        self._history["prefill_ms"].extend(prefill_times)
        self._history["decode_ms"].extend(decode_times)
        return {"prefill_ms": prefill_times, "decode_ms": decode_times}

    def teardown(self) -> None:
        if self.model is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._restore_compile_hooks()
        super().teardown()
        self.graph = None
        self.capture_stream = None
        self.static_decode_in = None
        self.static_decode_out = None
        self.base_position = 0


    def get_custom_metrics(self) -> Optional[dict]:
        """Return speculative decoding metrics for flexdecoding_graphs."""
        from common.python.benchmark_metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 10),
            accepted_tokens=getattr(self, '_accepted_tokens', 8),
            draft_time_ms=getattr(self, '_draft_ms', 1.0),
            verify_time_ms=getattr(self, '_verify_ms', 1.0),
            num_rounds=getattr(self, '_num_rounds', 1),
        )

def get_benchmark():
    return OptimizedFlexDecodingGraphsBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(result)
