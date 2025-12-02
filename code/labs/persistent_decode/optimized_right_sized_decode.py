"""Right-sized decode benchmark target with flag-based tier and quantization controls."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from typing import List, Tuple

from core.harness.benchmark_harness import BaseBenchmark
from labs.persistent_decode.optimized_persistent_decode_graphs import (
    GraphMode,
    OptimizedPersistentDecodeGraphsBenchmark,
)
from labs.persistent_decode.optimized_persistent_decode_triton import OptimizedPersistentDecodeTritonBenchmark
from labs.persistent_decode.persistent_decode_common import DecodeOptions, set_decode_options


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Right-sized decode benchmark flags", add_help=False)
    p.add_argument("--tier", choices=["small", "medium", "large"], default="medium", help="Decode tier sizing preset")
    p.add_argument(
        "--quantization",
        choices=["fp32", "fp16", "int4"],
        default="fp32",
        help="Decode quantization mode",
    )
    p.add_argument("--block-k", type=int, default=None, help="Override BLOCK_K (defaults depend on tier)")
    p.add_argument("--num-programs", type=int, default=None, help="Override number of Triton programs")
    p.add_argument(
        "--backend",
        choices=["triton", "graphs"],
        default="triton",
        help="Select the backend target for this lab",
    )
    p.add_argument(
        "--graph-mode",
        choices=[m.value for m in GraphMode],
        default=GraphMode.FULL_AND_PIECEWISE.value,
        help="Graph capture mode when using the graphs backend",
    )
    return p


def _parse_cli() -> Tuple[argparse.Namespace, List[str]]:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()
    return args, unknown


_CLI_ARGS, _CLI_UNKNOWN = _parse_cli()

set_decode_options(
    DecodeOptions(
        tier=_CLI_ARGS.tier,
        quantization=_CLI_ARGS.quantization,
        block_k=_CLI_ARGS.block_k,
        num_programs=_CLI_ARGS.num_programs,
    )
)


def get_benchmark() -> BaseBenchmark:
    if _CLI_ARGS.backend == "graphs":
        mode = GraphMode.from_str(_CLI_ARGS.graph_mode)
        return OptimizedPersistentDecodeGraphsBenchmark(graph_mode=mode)
    return OptimizedPersistentDecodeTritonBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
