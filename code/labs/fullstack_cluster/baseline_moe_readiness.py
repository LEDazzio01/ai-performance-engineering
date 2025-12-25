"""Baseline MoE readiness probe focused on all-to-all stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
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
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Baseline MoE readiness (all-to-all) probe")
    add_base_args(parser)
    parser.set_defaults(
        output_dir="artifacts/moe_readiness_baseline",
        message_sizes="16k,64k,256k,512k",
        skews="1.0,1.15,1.35",
        max_p99_ms=7.0,
        min_bw_gbps=12.0,
    )
    return parser


def _quick_preflight(rank: int) -> None:
    commands = [
        ("nvidia_topo", ["nvidia-smi", "topo", "-m"]),
        ("nic_caps", ["ibv_devinfo", "-v"]),
        ("gdr_modules", ["bash", "-lc", "lsmod | egrep 'nv_peer_mem|gdrdrv'"]),
    ]
    outputs = run_preflight(commands, rank=rank)
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
        _quick_preflight(rank)

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
    summarize_results(
        results,
        max_p99_ms=args.max_p99_ms,
        min_bw_gbps=args.min_bw_gbps,
        rank=rank,
        header="Baseline MoE all-to-all sweep",
    )


class BaselineMoEReadinessBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    def __init__(self) -> None:
        super().__init__()
        self._probe = torch.zeros(1, dtype=torch.float32)
        self._verify_output = torch.zeros(1, dtype=torch.float32)
        self.parameter_count = 0
        self.workload_size = 1
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._workload_registered = True

    def benchmark_fn(self) -> None:
        # On single-GPU hosts, skip rather than failing torchrun.
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: MoE readiness benchmark requires >=2 GPUs.")
        # Real work happens in the torchrun-launched script.
        self._verify_output = torch.zeros(1, device=self.device, dtype=torch.float32)

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=2,
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics for performance analysis."""
        # Basic metrics - override in subclass for domain-specific values
        return {
            "moe_readiness.workload_size": float(self.workload_size),
        }

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        script_path = Path(__file__).resolve()
        cfg = config or BenchmarkConfig()
        return TorchrunLaunchSpec(
            script_path=script_path,
            script_args=["--skip-preflight"],
            env={"NCCL_DEBUG": "WARN"},
            parse_rank0_only=True,
            multi_gpu_required=True,
            name="baseline_moe_readiness",
            config_arg_map={
                "iterations": "--iters",
            },
        )

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"probe": self._probe},
            output=self._verify_output,
            batch_size=1,
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )


def get_benchmark() -> BaselineMoEReadinessBenchmark:
    return BaselineMoEReadinessBenchmark()


if __name__ == "__main__":
    main()
