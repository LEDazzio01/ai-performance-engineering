"""Optimized MoE readiness probe with skew sweep, NCCL tweaks, and optional heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from labs.fullstack_cluster.moe_alltoall_common import (
    add_base_args,
    dtype_from_name,
    dump_json,
    init_distributed,
    parse_float_list,
    parse_size_list,
    run_alltoall_single,
    run_preflight,
    summarize_results,
    try_make_heatmaps,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Optimized MoE readiness (all-to-all) probe")
    add_base_args(parser)
    parser.set_defaults(
        message_sizes="32k,128k,512k,2m",
        skews="1.0,1.15,1.35",
        iters=30,
        max_p99_ms=5.5,
        min_bw_gbps=18.0,
        output_dir="artifacts/moe_readiness_optimized",
    )
    parser.add_argument("--disable-heatmaps", action="store_true", help="Skip matplotlib heatmap generation.")
    return parser


def _preflight(rank: int) -> None:
    commands = [
        ("nvidia_topo", ["nvidia-smi", "topo", "-m"]),
        ("nic_caps", ["ibv_devinfo", "-v"]),
        ("gdr_modules", ["bash", "-lc", "lsmod | egrep 'nv_peer_mem|gdrdrv'"]),
        ("rdma_counters", ["bash", "-lc", "perfquery -x -a 2>/dev/null || true"]),
        ("roce_stats", ["bash", "-lc", "ethtool -S eth0 2>/dev/null || true"]),
    ]
    outputs = run_preflight(commands, rank=rank, timeout_s=7.0)
    if rank == 0 and outputs:
        print("\n[Preflight]")
        for key, text in outputs.items():
            preview = text if text else "(no output)"
            print(f"{key}: {preview}")


def run_probe(args: argparse.Namespace) -> Tuple[Dict[Tuple[int, float], object], int, int]:
    rank, world_size, _, initialized = init_distributed()
    if world_size <= 1:
        print("Run with torchrun and at least 2 ranks to exercise all-to-all.")
        return {}, rank, world_size

    if not args.skip_preflight:
        _preflight(rank)

    dtype = dtype_from_name(args.dtype)
    sizes = parse_size_list(args.message_sizes)
    skews = parse_float_list(args.skews)

    results: Dict[Tuple[int, float], object] = {}
    for msg_bytes in sizes:
        for alpha in skews:
            if initialized:
                torch.distributed.barrier()
            res = run_alltoall_single(
                msg_bytes=msg_bytes,
                skew_alpha=alpha,
                num_iters=args.iters,
                world_size=world_size,
                dtype=dtype,
            )
            results[(msg_bytes, alpha)] = res

    if initialized:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return results, rank, world_size


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    results, rank, world_size = run_probe(args)
    if not results and world_size <= 1:
        return
    if not results:
        return

    output_dir = Path(args.output_dir)
    if rank == 0:
        dump_json(results, output_dir / "report.json", iters=args.iters, world_size=world_size)
        if not args.disable_heatmaps:
            written = try_make_heatmaps(results, output_dir, rank=rank)
            if written:
                print(f"Wrote heatmaps: {', '.join(str(p) for p in written)}")
    summarize_results(
        results,
        max_p99_ms=args.max_p99_ms,
        min_bw_gbps=args.min_bw_gbps,
        rank=rank,
        header="Optimized MoE all-to-all (skew-aware) sweep",
    )


class OptimizedMoEReadinessBenchmark(BaseBenchmark):
    """Harness entry that launches this module via torchrun with NCCL small-message knobs."""

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=4,
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=1200,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics for performance analysis."""
        # Basic metrics - override in subclass for domain-specific values
        return {
            "moe_readiness.workload_size": float(getattr(self, 'batch_size', 0)),
        }

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        script_path = Path(__file__).resolve()
        return TorchrunLaunchSpec(
            script_path=script_path,
            script_args=[],
            env={
                "NCCL_DEBUG": "WARN",
                "NCCL_PROTO": "LL128",
                "NCCL_CROSS_NIC": "1",
                "NCCL_MIN_NCHANNELS": "8",
                "NCCL_MAX_NCHANNELS": "32",
                "NCCL_IB_QPS_PER_CONNECTION": "2",
                "NCCL_NET_GDR_LEVEL": "PHB",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name="optimized_moe_readiness",
            config_arg_map={
                "iterations": "--iters",
            },
        )


def get_benchmark() -> OptimizedMoEReadinessBenchmark:
    return OptimizedMoEReadinessBenchmark()


if __name__ == "__main__":
    main()
