"""Baseline MoE all-to-all: fabric-agnostic NCCL env that underuses NVLink/IB."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.moe_parallelism.plan import (  # noqa: E402
    ClusterSpec,
    ModelSpec,
    ParallelismPlan,
    get_default_cluster_spec,
    get_default_model_spec,
)
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark  # noqa: E402

VALID_FABRICS = ("nvlink", "ib", "efa")


def _bytes_to_ms(value_bytes: float, bandwidth_GBps: float) -> float:
    if bandwidth_GBps <= 0.0:
        return float("inf")
    seconds = value_bytes / (bandwidth_GBps * (1024 ** 3))
    return seconds * 1e3


def _estimate_payload_bytes(plan: ParallelismPlan, model: ModelSpec) -> float:
    tokens_total = model.sequence_length * plan.microbatch_sequences
    moe_tokens_per_gpu = tokens_total * model.router_topk / max(plan.ep, 1)
    hidden_per_gpu = model.hidden_size / max(plan.tp, 1)
    return (
        moe_tokens_per_gpu
        * hidden_per_gpu
        * model.dtype_bytes
        * plan.microbatches
        * max(plan.ep - 1, 1)
    )


@dataclass(frozen=True)
class EnvProfile:
    name: str
    fabric: str
    efficiency: float
    env: Dict[str, str]
    notes: List[str]
    validation: List[str]


VALIDATION_COMMANDS = [
    "mpirun -np N --hostfile hosts -x NCCL_DEBUG=INFO ./build/alltoall_perf -b 1M -e 8G -f 2 -g 1",
    'NCCL_DEBUG=WARN TEST_DIR=./build TEST_ARGS="-dfloat -b8 -e32G -f2" $TEST_DIR/all_reduce_perf $TEST_ARGS',
]


BASELINE_PROFILES: Dict[str, EnvProfile] = {
    "nvlink": EnvProfile(
        name="NVLink (PXN/MNNVL disabled)",
        fabric="nvlink",
        efficiency=0.55,
        env={
            "NCCL_MNNVL_ENABLE": "0",
            "NCCL_PXN_DISABLE": "1",
            "NCCL_P2P_LEVEL": "PIX",
            "NCCL_PROTO": "Simple",
            "NCCL_NET_GDR_LEVEL": "SYS",
        },
        notes=[
            "Leaves the NVLink domain disabled so cross-GPU traffic falls back toward PCI/host paths.",
            "PXN is off, so NVLink+PCI striping never kicks in within a node.",
            "P2P level is PIX, which is safe for IB/EFA but wastes the NVSwitch fabric.",
        ],
        validation=VALIDATION_COMMANDS,
    ),
    "ib": EnvProfile(
        name="InfiniBand/EFA (NVLink-biased defaults)",
        fabric="ib",
        efficiency=0.65,
        env={
            "NCCL_MNNVL_ENABLE": "1",
            "NCCL_PXN_DISABLE": "0",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_PROTO": "LL128,Simple",
            "NCCL_NET_GDR_LEVEL": "0",
        },
        notes=[
            "Leaves NVLink-specific knobs on even though traffic rides the NICs.",
            "PXN on + NVL P2P cause NCCL to assume fast intra-node links that do not exist in this fabric.",
            "GDR level left at auto, risking mismatched HCAs when topology is shallow.",
        ],
        validation=VALIDATION_COMMANDS,
    ),
}

# EFA uses the same profile as InfiniBand unless the preset swaps NIC throughput.
BASELINE_PROFILES["efa"] = BASELINE_PROFILES["ib"]

CLUSTER: ClusterSpec = get_default_cluster_spec()
MODEL: ModelSpec = get_default_model_spec()


def build_plan(fabric: str = "nvlink") -> ParallelismPlan:
    cross_node = fabric != "nvlink"
    return ParallelismPlan(
        name=f"Baseline MoE all-to-all ({fabric})",
        dp=12,
        pp=3,
        tp=2,
        ep=8,
        microbatch_sequences=16,
        microbatches=10,
        experts_per_gpu=4,
        capacity_factor=1.15,
        dense_checkpoint_fraction=0.5,
        moe_checkpoint_fraction=0.75,
        stage_layers=[32, 32, 32],
        cross_node_ep=cross_node,
        notes=[
            "Topology-agnostic launch so all-to-all assumes whatever NCCL picks by default.",
            "Microbatches kept small so any latency spikes from fabric misuse are obvious.",
        ],
    )


def _render_summary(
    plan: ParallelismPlan,
    profile: EnvProfile,
    ep_ms: float,
    step_ms: float,
    throughput: float,
    payload_gb: float,
    effective_bw: float,
    base_bw: float,
) -> str:
    env_lines = [f"{k}={v}" for k, v in sorted(profile.env.items())]
    validation_lines = profile.validation or VALIDATION_COMMANDS
    note_lines = profile.notes or []
    lines = [
        f"{plan.name} — fabric: {profile.fabric} ({profile.name})",
        f"  Effective MoE all-to-all bandwidth: {effective_bw:.1f} GB/s "
        f"(base {base_bw:.1f} GB/s, efficiency {profile.efficiency:.2f})",
        f"  Payload ~{payload_gb:.2f} GB -> estimated all-to-all {ep_ms:.2f} ms",
        f"  Synthetic step: {step_ms:.1f} ms; projected throughput {throughput:,.0f} tokens/s",
        "  NCCL env (fabric-agnostic baseline):",
        indent("\n".join(env_lines), "    "),
        "  Validation commands:",
        indent("\n".join(validation_lines), "    "),
        "  Notes:",
        indent("\n".join(note_lines), "    "),
    ]
    return "\n".join(lines)


class BaselineAllToAllEnvBenchmark(PlanBenchmark):
    """Baseline NCCL env that ignores NVLink vs IB/EFA nuances."""

    def __init__(self, fabric: str = "nvlink") -> None:
        fabric_key = fabric if fabric in BASELINE_PROFILES else "nvlink"
        self.profile = BASELINE_PROFILES[fabric_key]
        plan = build_plan(fabric_key)
        super().__init__(plan, CLUSTER, MODEL)
        self.step_ms: float = 0.0
        self.throughput: float = 0.0
        self.effective_bw: float = 0.0
        self.payload_gb: float = 0.0

    def benchmark_fn(self) -> None:
        report = self.evaluator.analyze(self.plan)
        payload_bytes = _estimate_payload_bytes(self.plan, self.model)
        base_bw = (
            self.cluster.nvlink_bandwidth_GBps
            if self.profile.fabric == "nvlink"
            else self.cluster.nic_bandwidth_GBps
        )
        self.effective_bw = base_bw * self.profile.efficiency
        ep_ms = _bytes_to_ms(payload_bytes, self.effective_bw)
        self.payload_gb = payload_bytes / (1024 ** 3)
        self.step_ms = (
            report.compute_ms + report.dp_time_ms + report.pipeline_time_ms + ep_ms
        )
        global_batch_tokens = (
            self.model.sequence_length
            * self.plan.microbatch_sequences
            * self.plan.microbatches
            * self.plan.dp
        )
        self.throughput = (
            global_batch_tokens / (self.step_ms / 1e3) if self.step_ms > 0 else 0.0
        )
        self.report = report
        self._summary = _render_summary(
            self.plan,
            self.profile,
            ep_ms,
            self.step_ms,
            self.throughput,
            self.payload_gb,
            self.effective_bw,
            base_bw,
        )
        self._finalize_output([self.step_ms, self.throughput, self.payload_gb, self.effective_bw])

    def print_summary(self) -> None:
        if self._summary:
            print(self._summary)

    def teardown(self) -> None:
        # Keep report/summary for post-run printing in __main__.
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fabric",
        choices=VALID_FABRICS,
        default=os.environ.get("MPE_FABRIC", "nvlink"),
        help="Fabric to model (nvlink vs ib/efa).",
    )
    return parser.parse_args()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    fabric = os.environ.get("MPE_FABRIC", "nvlink").lower()
    if fabric not in VALID_FABRICS:
        fabric = "nvlink"
    return BaselineAllToAllEnvBenchmark(fabric=fabric)


if __name__ == "__main__":
    args = _parse_args()
    benchmark = BaselineAllToAllEnvBenchmark(fabric=args.fabric)
    run_benchmark(benchmark)
    benchmark.print_summary()
