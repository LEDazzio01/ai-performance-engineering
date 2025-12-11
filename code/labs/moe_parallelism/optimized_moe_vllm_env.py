"""Optimized vLLM MoE environment: NVLink vs NVL72 presets (LL/LL128 + graphs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.common.model_fetcher import ensure_gpt_oss_20b  # noqa: E402
from labs.moe_parallelism.baseline_moe_vllm_env import (  # noqa: E402
    MODEL,
    NVL72_CLUSTER,
    NVLINK_CLUSTER,
    VALID_FABRICS,
    build_plan,
    _default_run_args,
    _build_arg_parser,
    TARGET_NAME,
)
from labs.moe_parallelism.benchmarking import PlanBenchmark, run_benchmark  # noqa: E402
from labs.moe_parallelism.moe_env_presets import (  # noqa: E402
    EnvPreset,
    estimate_collective,
    render_summary,
)
from labs.moe_parallelism.plan import format_report  # noqa: E402


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


def _optimized_presets() -> Dict[str, EnvPreset]:
    nvlink_env = EnvPreset(
        name="NVLink MoE (LL/LL128 + graphs)",
        fabric="nvlink",
        efficiency=0.93,
        handshake_ms=1.0,
        env={
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUDA_GRAPH_CACHE_SIZE": "4194304",
            "NCCL_P2P_LEVEL": "SYS",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "2",
            "NCCL_PROTO": "LL,LL128,Simple",
            "NCCL_ALGO": "Tree,Ring",
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_MAX_NCHANNELS": "16",
            "NCCL_NTHREADS": "512",
            "NCCL_LL_THRESHOLD": "2048",
            "NCCL_LL128_THRESHOLD": "16384",
            "NCCL_TOPO_DUMP_FILE": "/tmp/nccl_topo.xml",
            "NCCL_GRAPH_DUMP_FILE": "/tmp/nccl_graph.xml",
            "NCCL_IB_DISABLE": "1",
            "NCCL_LAUNCH_MODE": "GROUP",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_HIGH_PRIORITY": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_CUDA_GRAPH": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        },
        notes=[
            "Forces NVLink/NVSwitch by disabling IB and raising P2P to SYS.",
            "LL/LL128 + Tree/Ring keep per-token all-to-all latency low; channel count lets many tiny MoE shards overlap.",
            "CUDA graph cache + spawn worker method reduce handshake stalls in decode loops.",
            "Pin CUDA_VISIBLE_DEVICES to a single NVSwitch island (0-7 by default) when running on larger boxes.",
        ],
        validation=[
            "NVBandwidth -gpu 0-7 -bidirectional -csv > /tmp/nvbandwidth.csv",
            "mpirun -np {ngpu} --bind-to none all_reduce_perf -b 512 -e 64K -f 2 -g 1 -c 1 -n 200",
            "mpirun -np {ngpu} --bind-to none alltoall_perf  -b 512 -e 64K -f 2 -g 1 -c 1 -n 200",
        ],
    )

    nvl72_env = EnvPreset(
        name="NVL72/IB MoE (LL128 + dual-rail QPs)",
        fabric="nvl72",
        efficiency=0.82,
        handshake_ms=1.5,
        env={
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUDA_GRAPH_CACHE_SIZE": "4194304",
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "2",
            "NCCL_IB_GID_INDEX": "0",
            "NCCL_IB_PCI_RELAXED_ORDERING": "1",
            "NCCL_IB_QPS_PER_CONNECTION": "2",
            "NCCL_CROSS_NIC": "1",
            "NCCL_SOCKET_NTHREADS": "4",
            "NCCL_NSOCKS_PERTHREAD": "2",
            "NCCL_PROTO": "LL128,LL,Simple",
            "NCCL_ALGO": "Tree,Ring",
            "NCCL_MIN_NCHANNELS": "8",
            "NCCL_MAX_NCHANNELS": "32",
            "NCCL_NTHREADS": "512",
            "NCCL_LL_THRESHOLD": "4096",
            "NCCL_LL128_THRESHOLD": "32768",
            "NCCL_NET_PROTOCOL": "SIMPLE",
            "NCCL_IB_SL": "0",
            "NCCL_IB_TC": "106",
            "NCCL_LAUNCH_MODE": "GROUP",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_HIGH_PRIORITY": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_CUDA_GRAPH": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        },
        notes=[
            "LL128-first + Tree/Ring keep small cross-node payloads predictable on IB.",
            "Two QPs/connection and CROSS_NIC=1 reduce head-of-line blocking on NVL72 dual-rail fabrics.",
            "Simple net protocol keeps congestion behavior stable; TC=106 is ECN-friendly (tune to your fabric).",
            "Keep GID/DSCP aligned with cluster policy; flip IB_SL/TC accordingly.",
        ],
        validation=[
            "mpirun -np {ngpu} --bind-to none all_reduce_perf -b 512 -e 64K -f 2 -g 1 -c 1 -n 200",
            "mpirun -np {ngpu} --bind-to none alltoall_perf  -b 512 -e 64K -f 2 -g 1 -c 1 -n 200",
            "ENABLE_NVLINK=0 nvsysinfo || true",
        ],
    )
    return {"nvlink": nvlink_env, "nvl72": nvl72_env}


OPTIMIZED_PRESETS = _optimized_presets()


class OptimizedMoeVllmEnvBenchmark(PlanBenchmark):
    """Fabric-aware NCCL/vLLM env tuned for MoE inference."""

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
        fabric_key = fabric if fabric in OPTIMIZED_PRESETS else "nvlink"
        self.preset = OPTIMIZED_PRESETS[fabric_key]
        self.fabric = fabric_key
        cluster = NVL72_CLUSTER if fabric_key == "nvl72" else NVLINK_CLUSTER
        plan = build_plan("Optimized", fabric_key)
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
    return OptimizedMoeVllmEnvBenchmark()


if __name__ == "__main__":
    args = _parse_args()
    bench = OptimizedMoeVllmEnvBenchmark(
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
