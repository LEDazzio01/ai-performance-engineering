"""Optimized decode loop with bucketed shapes, preallocated workspaces, and lazy KV compaction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

# Workaround for importlib.util.spec_from_file_location loading:
# Register this module in sys.modules so @dataclass works correctly
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from dataclasses import dataclass

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Handle import robustly - baseline module may not be in sys.modules yet
try:
    from ch18.baseline_vllm_decode_graphs import (  # noqa: E402
        DecodeMetrics,
        default_trace,
        export_prom_metrics,
        format_metrics,
    )
except ImportError:
    # Fallback: load baseline module directly
    import importlib.util
    _baseline_path = Path(__file__).parent / "baseline_vllm_decode_graphs.py"
    _spec = importlib.util.spec_from_file_location("ch18.baseline_vllm_decode_graphs", _baseline_path)
    _baseline_module = importlib.util.module_from_spec(_spec)
    sys.modules["ch18.baseline_vllm_decode_graphs"] = _baseline_module
    _spec.loader.exec_module(_baseline_module)
    DecodeMetrics = _baseline_module.DecodeMetrics
    default_trace = _baseline_module.default_trace
    export_prom_metrics = _baseline_module.export_prom_metrics
    format_metrics = _baseline_module.format_metrics

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from common.python.smoke import is_smoke_mode  # noqa: E402
from ch18.decode_kernels import DEVICE, build_decode_kernel  # noqa: E402

BUCKETS = (8, 16, 24, 32)
FRAG_LIMIT = 0.20
AGE_LIMIT = 6  # steps; small for the toy demo


def pick_bucket(size: int) -> int:
    for b in BUCKETS:
        if size <= b:
            return b
    return BUCKETS[-1]


def pad_to_bucket(tensor: torch.Tensor, bucket: int) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Pad to the bucket size and return a mask for the real rows."""
    if tensor.size(0) == bucket:
        return tensor, None
    pad = bucket - tensor.size(0)
    padding = torch.empty((pad, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    mask = torch.ones(bucket, dtype=torch.bool, device=tensor.device)
    mask[tensor.size(0) :] = False
    return torch.cat([tensor, padding], dim=0), mask


@dataclass
class BucketWorkspace:
    batch: int
    hidden: int
    device: str = DEVICE
    tokens: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    tmp: torch.Tensor | None = None
    stream: torch.cuda.Stream | None = None
    initialized: bool = False

    def ensure(self) -> None:
        if self.initialized:
            return
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokens = torch.empty((self.batch, self.hidden), device=self.device, dtype=dtype)
        self.kv = torch.empty_like(self.tokens)
        self.mask = torch.ones(self.batch, dtype=torch.bool, device=self.device)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                self.logits = torch.empty((self.batch, self.hidden), device=self.device, dtype=dtype)
                self.tmp = torch.empty_like(self.logits)
            self.stream.synchronize()
        else:
            self.logits = torch.empty((self.batch, self.hidden), device=self.device, dtype=dtype)
            self.tmp = torch.empty_like(self.logits)
        self.initialized = True

    @property
    def bytes(self) -> int:
        if self.logits is None or self.tmp is None or self.tokens is None or self.kv is None or self.mask is None:
            return 0
        return (
            (self.logits.numel() + self.tmp.numel()) * self.logits.element_size()
            + self.tokens.numel() * self.tokens.element_size()
            + self.kv.numel() * self.kv.element_size()
            + self.mask.numel() * self.mask.element_size()
        )


@dataclass
class KVBlock:
    capacity: int
    used: int = 0
    age: int = 0

    @property
    def free_ratio(self) -> float:
        if self.capacity == 0:
            return 0.0
        return max(0.0, 1.0 - (self.used / self.capacity))

    def advance(self, tokens_written: int) -> None:
        self.used = min(self.capacity, self.used + tokens_written)
        self.age += 1

    def maybe_compact(self) -> bool:
        if self.free_ratio > FRAG_LIMIT and self.age > AGE_LIMIT:
            self.used = 0
            self.age = 0
            return True
        return False


class LazyKVLayout:
    """Defers compaction until fragmentation and age cross a threshold."""

    def __init__(self, blocks: Iterable[int] = (96, 128, 160)) -> None:
        self.blocks = [KVBlock(b) for b in blocks]

    def simulate_step(self, tokens_written: int) -> int:
        compactions = 0
        for block in self.blocks:
            block.advance(tokens_written)
            if block.maybe_compact():
                compactions += 1
        return compactions


class OptimizedDecodeDriver:
    """
    Decode loop that keeps shapes stable, preallocates per-bucket workspaces, and lazily compacts KV.
    """

    def __init__(self, trace: Iterable[int] | None = None, hidden: int = 256) -> None:
        self.trace = list(trace) if trace is not None else default_trace()
        self.hidden = hidden
        self.decode_kernel = build_decode_kernel(hidden=self.hidden, max_batch=max(BUCKETS))
        self.kv_layout = LazyKVLayout()
        self.workspaces: Dict[int, BucketWorkspace] = {}
        self.captured_shapes: set[Tuple[int, int]] = set()

    def workspace_for(self, bucket: int) -> BucketWorkspace:
        if bucket not in self.workspaces:
            self.workspaces[bucket] = BucketWorkspace(batch=bucket, hidden=self.hidden)
        return self.workspaces[bucket]

    def run(self) -> DecodeMetrics:
        metrics = DecodeMetrics()
        dtype = torch.float16 if getattr(self.decode_kernel, "backend", "") == "vllm" else torch.float32
        for batch_size in self.trace:
            bucket = pick_bucket(batch_size)
            ws = self.workspace_for(bucket)
            was_initialized = ws.initialized
            ws.ensure()

            if ws.tokens is None or ws.kv is None or ws.mask is None:
                raise RuntimeError("workspace not initialized")
            # Populate preallocated padded buffers in-place to keep shapes stable.
            ws.tokens.normal_(mean=0.0, std=1.0)
            ws.kv.normal_(mean=0.0, std=1.0)
            logits = self.decode_kernel(ws.tokens, ws.kv, ws.mask)

            shape_key = (logits.shape[0], logits.shape[1])
            if shape_key not in self.captured_shapes:
                metrics.graph_recaptures += 1
                self.captured_shapes.add(shape_key)

            # Count allocator growth only once per workspace instead of every step.
            metrics.allocator_bytes += 0 if was_initialized else ws.bytes
            metrics.compactions += self.kv_layout.simulate_step(tokens_written=batch_size)
            metrics.tokens += batch_size
            metrics.steps += 1

        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucketed decode loop with reusable workspaces.")
    parser.add_argument("--steps", type=int, default=24, help="Decode iterations to run.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size for the mock decode op.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the ragged trace.")
    parser.add_argument("--prom-port", type=int, default=None, help="Optional Prometheus port to export metrics.")
    parser.add_argument(
        "--prom-duration",
        type=int,
        default=0,
        help="If --prom-port is set, keep the server alive this many seconds (0 = fire-and-exit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = default_trace(num_steps=args.steps, seed=args.seed)
    driver = OptimizedDecodeDriver(trace=trace, hidden=args.hidden)
    metrics = driver.run()
    backend = getattr(driver.decode_kernel, "backend", "torch")
    print(format_metrics("optimized", metrics, backend=backend))

    if args.prom_port is not None:
        export_prom_metrics("optimized", metrics, backend=backend, port=args.prom_port, duration_s=args.prom_duration)


class OptimizedVLLMDecodeGraphsBenchmark(BaseBenchmark):
    """
    Harness entry point for the bucketed decode loop. Uses the same ragged
    trace as the baseline so speedups reflect graph reuse and allocator reuse,
    not synthetic workload changes.
    """

    def __init__(self, steps: int = 32, hidden: int = 192, seed: int = 0) -> None:
        if is_smoke_mode():
            raise RuntimeError("SKIPPED: vllm_decode_graphs is disabled in smoke-test sweeps")
        super().__init__()
        self.steps = steps
        self.hidden = hidden
        self.seed = seed
        self._trace: list[int] = default_trace(num_steps=self.steps, seed=self.seed)
        self._driver: Optional[OptimizedDecodeDriver] = None
        self._last_metrics: Optional[DecodeMetrics] = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            percentiles=[50, 75, 90, 99],
        )

    def setup(self) -> None:
        torch.manual_seed(self.seed)
        self._driver = OptimizedDecodeDriver(trace=self._trace, hidden=self.hidden)

    def benchmark_fn(self) -> None:
        if self._driver is None:
            raise RuntimeError("SKIPPED: optimized decode driver not initialized")
        torch.cuda.synchronize()
        self._last_metrics = self._driver.run()
        torch.cuda.synchronize()

    def teardown(self) -> None:
        super().teardown()
        self._driver = None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self._last_metrics is None:
            return None
        return {
            "vllm_decode_graphs.steps": float(self._last_metrics.steps),
            "vllm_decode_graphs.tokens": float(self._last_metrics.tokens),
            "vllm_decode_graphs.graph_recaptures": float(self._last_metrics.graph_recaptures),
            "vllm_decode_graphs.allocator_mb": float(self._last_metrics.allocator_bytes) / (1024 * 1024),
            "vllm_decode_graphs.compactions": float(self._last_metrics.compactions),
        }


def get_benchmark() -> BaseBenchmark:
    return OptimizedVLLMDecodeGraphsBenchmark()


if __name__ == "__main__":
    main()
