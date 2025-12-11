"""Baseline vLLM MoE environment: untuned small-message collectives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.common.model_fetcher import ensure_gpt_oss_20b  # noqa: E402
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark  # noqa: E402
from labs.moe_parallelism.moe_env_presets import (  # noqa: E402
    DEFAULT_VALIDATION,
    EnvPreset,
    estimate_collective,
    render_summary,
)
from labs.moe_parallelism.plan import (  # noqa: E402
    ClusterSpec,
    ModelSpec,
    ParallelismPlan,
    format_report,
    get_default_model_spec,
)

VALID_FABRICS = ("nvlink", "nvl72")
TARGET_NAME = "moe_vllm_env"

NVLINK_CLUSTER = ClusterSpec(
    name="GB200/B200 NVLink (single node)",
    nodes=1,
    gpus_per_node=8,
    nics_per_node=2,
    nic_bandwidth_gbps=800.0,
    nvlink_bandwidth_tbps=1.8,
    hbm_gb=192.0,
    interconnect="NVLink",
)

NVL72_CLUSTER = ClusterSpec(
    name="GB200 NVL72 (dual-rail IB)",
    nodes=9,
    gpus_per_node=8,
    nics_per_node=2,
    nic_bandwidth_gbps=800.0,
    nvlink_bandwidth_tbps=1.8,
    hbm_gb=192.0,
    interconnect="InfiniBand",
)

MODEL: ModelSpec = get_default_model_spec()


def _baseline_presets() -> Dict[str, EnvPreset]:
    nvlink_env = EnvPreset(
        name="Untuned NVLink (defaults, NICs left on)",
        fabric="nvlink",
        efficiency=0.58,
        handshake_ms=4.0,
        env={
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
            "NCCL_P2P_LEVEL": "PIX",
            "NCCL_PROTO": "Simple",
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "1",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_NTHREADS": "256",
            "NCCL_LL_THRESHOLD": "65536",
            "NCCL_LL128_THRESHOLD": "131072",
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "0",
            "NCCL_LAUNCH_MODE": "PARALLEL",
            "NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_HIGH_PRIORITY": "0",
            "VLLM_USE_CUDA_GRAPH": "0",
        },
        notes=[
            "Leaves NICs enabled and PIX P2P, so some MoE shards spill over PCIe/NIC paths.",
            "Defaults keep LL/LL128 disabled for small messages; routers sit on Simple/Ring handshakes.",
            "No CUDA graph capture or high-priority NCCL streams, so decode sees extra launch jitter.",
        ],
        validation=DEFAULT_VALIDATION,
    )

    nvl72_env = EnvPreset(
        name="Untuned NVL72/IB (socket defaults)",
        fabric="nvl72",
        efficiency=0.45,
        handshake_ms=10.0,
        env={
            "CUDA_DEVICE_MAX_CONNECTIONS": "8",
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "0",
            "NCCL_IB_QPS_PER_CONNECTION": "1",
            "NCCL_CROSS_NIC": "0",
            "NCCL_SOCKET_NTHREADS": "1",
            "NCCL_NSOCKS_PERTHREAD": "1",
            "NCCL_PROTO": "Simple",
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "1",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_LL_THRESHOLD": "65536",
            "NCCL_LL128_THRESHOLD": "131072",
            "NCCL_NET_PROTOCOL": "Socket",
            "NCCL_IB_TC": "0",
            "NCCL_LAUNCH_MODE": "PARALLEL",
            "NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_HIGH_PRIORITY": "0",
            "VLLM_USE_CUDA_GRAPH": "0",
        },
        notes=[
            "Relies on default socket paths: one QP, no cross-NIC striping, and large LL thresholds.",
            "No RoCE/IB traffic class guidance; congestion can worsen MoE p99 latencies.",
            "Keeps graph capture off, so router/expert bursts repeatedly negotiate streams.",
        ],
        validation=DEFAULT_VALIDATION,
    )
    return {"nvlink": nvlink_env, "nvl72": nvl72_env}


BASELINE_PRESETS = _baseline_presets()


def _default_run_args(fabric: str) -> Dict[str, object]:
    if fabric == "nvl72":
        return {
            "ngpu": 72,
            "tp": 8,
            "pp": 3,
            "max_len": 8192,
            "max_seqs": 4096,
            "gpu_util": 0.92,
            "model_path": "gpt-oss-20b/original/",
        }
    return {
        "ngpu": 8,
        "tp": 8,
        "pp": 1,
        "max_len": 8192,
        "max_seqs": 1024,
        "gpu_util": 0.92,
        "model_path": "gpt-oss-20b/original/",
    }


def _build_arg_parser(default_fabric: str = "nvlink") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fabric",
        choices=VALID_FABRICS,
        default=default_fabric,
        help="Fabric to model: nvlink (single node) or nvl72 (multi-node InfiniBand).",
    )
    parser.add_argument("--ngpu", type=int, help="Number of GPUs to use in validation commands.")
    parser.add_argument("--model", dest="model_path", default=None, help="Model path for validation.")
    parser.add_argument("--tp", type=int, help="Tensor parallel degree for validation commands.")
    parser.add_argument("--pp", type=int, help="Pipeline parallel degree for validation commands.")
    parser.add_argument("--max-len", type=int, dest="max_len", help="Max sequence length for validation commands.")
    parser.add_argument("--max-seqs", type=int, dest="max_seqs", help="Max concurrent sequences for validation commands.")
    parser.add_argument("--gpu-util", type=float, dest="gpu_util", help="GPU memory utilization target for vLLM.")
    return parser


def build_plan(label: str, fabric: str) -> ParallelismPlan:
    cross_node = fabric == "nvl72"
    if fabric == "nvlink":
        return ParallelismPlan(
            name=f"{label} vLLM MoE env ({fabric})",
            dp=1,
            pp=1,
            tp=2,
            ep=4,
            microbatch_sequences=8,
            microbatches=6,
            experts_per_gpu=4,
            capacity_factor=1.15,
            dense_checkpoint_fraction=0.75,
            moe_checkpoint_fraction=0.9,
            stage_layers=None,
            cross_node_ep=False,
            notes=[
                "Single-node decode with top-2 routing; small microbatches keep MoE all-to-all visible.",
                "No cross-node experts; expect NVLink-only traffic when tuned correctly.",
            ],
        )

    return ParallelismPlan(
        name=f"{label} vLLM MoE env ({fabric})",
        dp=3,
        pp=3,
        tp=2,
        ep=4,
        microbatch_sequences=8,
        microbatches=8,
        experts_per_gpu=4,
        capacity_factor=1.15,
        dense_checkpoint_fraction=0.75,
        moe_checkpoint_fraction=0.9,
        stage_layers=None,
        cross_node_ep=cross_node,
        notes=[
            "Nine-node NVL72-style layout; experts span nodes so all-to-all rides IB.",
            "Router-heavy microbatches surface small-message latency and QP congestion.",
        ],
    )


class BaselineMoeVllmEnvBenchmark(PlanBenchmark):
    def __init__(
        self,
        fabric: str = "nvlink",
        ngpu: Optional[int] = None,
        model_path: Optional[str] = None,
        tp: Optional[int] = None,
        pp: Optional[int] = None,
        max_len: Optional[int] = None,
        max_seqs: Optional[int] = None,
        gpu_util: Optional[float] = None,
    ) -> None:
        fabric_key = fabric if fabric in BASELINE_PRESETS else "nvlink"
        self.preset = BASELINE_PRESETS[fabric_key]
        self.fabric = fabric_key
        cluster = NVL72_CLUSTER if fabric_key == "nvl72" else NVLINK_CLUSTER
        plan = build_plan("Baseline", fabric_key)
        super().__init__(plan, cluster=cluster, model=MODEL)
        self.payload_bytes = 0.0
        self.base_bw = 0.0
        self.effective_bw = 0.0
        self.collective_ms = 0.0
        self.step_ms = 0.0
        self.throughput = 0.0
        defaults = _default_run_args(fabric_key)
        self.ngpu = int(ngpu or defaults["ngpu"])
        self.model_path = str(model_path or defaults["model_path"])
        ensure_gpt_oss_20b(Path(self.model_path))
        self.tp = int(tp or defaults["tp"])
        self.pp = int(pp or defaults["pp"])
        self.max_len = int(max_len or defaults["max_len"])
        self.max_seqs = int(max_seqs or defaults["max_seqs"])
        self.gpu_util = float(gpu_util or defaults["gpu_util"])
        self._overrides_applied = False

    def benchmark_fn(self) -> None:
        self._apply_target_overrides()
        report = self.evaluator.analyze(self.plan)
        payload_bytes, base_bw, effective_bw, collective_ms = estimate_collective(
            self.plan,
            MODEL,
            self.cluster,
            self.preset.efficiency,
            self.preset.handshake_ms,
        )
        step_ms = (
            report.compute_ms + report.dp_time_ms + report.pipeline_time_ms + collective_ms
        )
        global_batch_tokens = (
            MODEL.sequence_length
            * self.plan.microbatch_sequences
            * self.plan.microbatches
            * self.plan.dp
        )
        throughput = global_batch_tokens / (step_ms / 1e3) if step_ms > 0 else 0.0

        self.payload_bytes = payload_bytes
        self.base_bw = base_bw
        self.effective_bw = effective_bw
        self.collective_ms = collective_ms
        self.step_ms = step_ms
        self.throughput = throughput
        self.report = report
        self._summary = render_summary(
            self.plan,
            self.preset,
            report=format_report(report),
            payload_bytes=payload_bytes,
            base_bw=base_bw,
            effective_bw=effective_bw,
            collective_ms=collective_ms,
            step_ms=step_ms,
            throughput=throughput,
            format_kwargs={
                "ngpu": self.ngpu,
                "model": self.model_path,
                "tp": self.tp,
                "pp": self.pp,
                "max_len": self.max_len,
                "max_seqs": self.max_seqs,
                "gpu_util": self.gpu_util,
            },
        )
        self._finalize_output([self.step_ms, self.throughput, self.payload_bytes, self.effective_bw])

    def _apply_target_overrides(self) -> None:
        if self._overrides_applied:
            return
        cfg = getattr(self, "_config", None)
        if not cfg:
            return
        overrides_map = getattr(cfg, "target_extra_args", {}) or {}
        label = getattr(cfg, "target_label", None)
        candidates = [label] if label else []
        if label and ":" in label:
            candidates.append(label.split(":", 1)[1])
        candidates.append(TARGET_NAME)
        arg_list: Optional[list[str]] = None
        for key in candidates:
            if key and key in overrides_map:
                value = overrides_map[key]
                if isinstance(value, str):
                    arg_list = value.split()
                else:
                    arg_list = list(value)
                break
        if not arg_list:
            self._overrides_applied = True
            return
        parser = _build_arg_parser(default_fabric=self.fabric)
        parsed, _ = parser.parse_known_args(arg_list)
        defaults = _default_run_args(parsed.fabric)
        self.ngpu = int(parsed.ngpu or defaults["ngpu"])
        self.model_path = str(parsed.model_path or defaults["model_path"])
        self.tp = int(parsed.tp or defaults["tp"])
        self.pp = int(parsed.pp or defaults["pp"])
        self.max_len = int(parsed.max_len or defaults["max_len"])
        self.max_seqs = int(parsed.max_seqs or defaults["max_seqs"])
        self.gpu_util = float(parsed.gpu_util or defaults["gpu_util"])
        self._overrides_applied = True


def _parse_args() -> argparse.Namespace:
    parser = _build_arg_parser()
    return parser.parse_args()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> "PlanBenchmark":
    from labs.moe_parallelism.benchmarking import is_plan_available, get_skip_benchmark
    if not is_plan_available():
        return get_skip_benchmark()
    return BaselineMoeVllmEnvBenchmark()


if __name__ == "__main__":
    args = _parse_args()
    bench = BaselineMoeVllmEnvBenchmark(
        fabric=args.fabric,
        ngpu=args.ngpu,
        model_path=args.model_path,
        tp=args.tp,
        pp=args.pp,
        max_len=args.max_len,
        max_seqs=args.max_seqs,
        gpu_util=args.gpu_util,
    )
    run_benchmark(bench)
    bench.print_summary()
