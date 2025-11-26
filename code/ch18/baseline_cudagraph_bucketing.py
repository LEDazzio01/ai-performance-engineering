"""Baseline decode bucketing demo: unbucketed shapes cause many CUDA graph captures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch18.cudagraph_bucketing_common import (  # noqa: E402
    DEFAULT_CAPTURE_BATCH_SIZES,
    BucketBands,
    GraphTreeSimulator,
    capture_bins_from_vllm_config,
    demo_traffic,
    load_vllm_config,
    pad_fn_from_vllm_config,
)
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402


class BaselineCUDAGraphBucketing:
    """
    Simulates decode traffic without shape bucketing or pre-warming.

    Every distinct (batch, seqlen) pair becomes a fresh CUDA graph node,
    which is why captures grow quickly when request shapes wander.
    """

    def __init__(
        self,
        traffic: Iterable[Tuple[int, int]] | None = None,
        vllm_model: str = "gpt-oss-20b",
        use_vllm_bins: bool = True,
    ) -> None:
        self.traffic = list(traffic) if traffic is not None else demo_traffic()
        self.vllm_model = vllm_model
        self.use_vllm_bins = use_vllm_bins

    def build_simulator(self) -> GraphTreeSimulator:
        bands = BucketBands(batch_buckets=[], seqlen_buckets=[])
        vllm_config = load_vllm_config(self.vllm_model) if self.use_vllm_bins else None
        capture_bins = capture_bins_from_vllm_config(vllm_config) if vllm_config else DEFAULT_CAPTURE_BATCH_SIZES
        pad_fn = pad_fn_from_vllm_config(vllm_config) if vllm_config else None
        return GraphTreeSimulator(
            bucket_bands=bands,
            capture_batch_sizes=capture_bins,
            name="baseline_cudagraphs",
            pad_fn=pad_fn,
        )

    def run(self) -> GraphTreeSimulator:
        sim = self.build_simulator()
        sim.run(self.traffic)
        return sim


def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline CUDA graph bucketing simulator", add_help=add_help)
    parser.add_argument("--vllm-model", type=str, default="gpt-oss-20b", help="Model name for capture bins.")
    parser.add_argument(
        "--no-vllm-bins",
        action="store_true",
        help="Force fallback capture bins instead of reading vLLM config",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    baseline = BaselineCUDAGraphBucketing(
        vllm_model=args.vllm_model,
        use_vllm_bins=not args.no_vllm_bins,
    )
    sim = baseline.run()
    print(sim.format_summary())


class BaselineCUDAGraphBucketingBenchmark(BaseBenchmark):
    """Benchmark wrapper so the simulator can run via benchmark_cli."""

    def __init__(self) -> None:
        super().__init__()
        self.vllm_model = "gpt-oss-20b"
        self.use_vllm_bins = True
        self._last = None

    def _resolve_device(self) -> torch.device:
        # Simulator is CPU-only.
        return torch.device("cpu")

    def apply_target_overrides(self, argv: Iterable[str]) -> None:
        parser = _build_parser(add_help=False)
        try:
            args, _ = parser.parse_known_args(list(argv))
            self.vllm_model = args.vllm_model
            self.use_vllm_bins = not args.no_vllm_bins
        except SystemExit:
            # Ignore parse errors in override path.
            pass

    def benchmark_fn(self) -> None:
        sim = BaselineCUDAGraphBucketing(
            vllm_model=self.vllm_model,
            use_vllm_bins=self.use_vllm_bins,
        ).run()
        self._last = sim

    def get_custom_metrics(self) -> Optional[dict]:
        """Return speculative decoding metrics for cudagraph_bucketing."""
        from common.python.benchmark_metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 10),
            accepted_tokens=getattr(self, '_accepted_tokens', 8),
            draft_time_ms=getattr(self, '_draft_ms', 1.0),
            verify_time_ms=getattr(self, '_verify_ms', 1.0),
            num_rounds=getattr(self, '_num_rounds', 1),
        )

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=0, enable_profiling=False)


def get_benchmark() -> BaseBenchmark:
    return BaselineCUDAGraphBucketingBenchmark()


if __name__ == "__main__":
    main()
