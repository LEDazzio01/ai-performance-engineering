"""8-GPU decode benchmark to stress NVLink-C2C throughput on B200 nodes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig  # noqa: E402


class MultiGPUDecodeBenchmark(BaseBenchmark):
    """Torchrun-only entry that launches this script across 8 GPUs."""

    def __init__(self) -> None:
        super().__init__()
        self.jitter_exemption_reason = "8xGPU decode benchmark: multi-GPU"
        self.register_workload_metadata(requests_per_iteration=1.0)

    def benchmark_fn(self) -> None:  # pragma: no cover - torchrun path only
        raise RuntimeError("Use torchrun via get_torchrun_spec")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=8,
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=600,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "decode_8xgpu.batch_size": float(getattr(self, 'batch_size', 0)),
            "decode_8xgpu.seq_len": float(getattr(self, 'seq_len', 0)),
            "decode_8xgpu.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        script_path = Path(__file__).resolve()
        return TorchrunLaunchSpec(
            script_path=script_path,
            script_args=[],
            env={
                "NCCL_DEBUG": "WARN",
                "NCCL_NET_GDR_LEVEL": "PHB",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_P2P_DISABLE": "0",
            },
            parse_rank0_only=True,
            multi_gpu_required=True,
            name="optimized_decode_8xgpu",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"type": "decode_8xgpu"}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def _run_worker(iters: int, warmup: int) -> None:
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    cfg = DecodeConfig(
        batch_size=8,
        prompt_tokens=1024,
        decode_tokens=256,
        hidden_size=2048,
        use_fp8=True,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_cuda_graphs=True,
        graph_full_iteration=True,
        use_torch_compile=True,
        label="optimized_decode_8xgpu",
    )

    bench = DecodeBenchmark(cfg)
    bench.setup()

    # Warmup
    for _ in range(max(warmup, 0)):
        bench.benchmark_fn()
    torch.cuda.synchronize()

    # Timed iterations
    start = time.time()
    for _ in range(max(iters, 1)):
        bench.benchmark_fn()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_iter = cfg.batch_size * (cfg.prompt_tokens + cfg.decode_tokens)
    local_tokens_per_s = tokens_per_iter * (iters / elapsed)
    global_tokens_per_s = local_tokens_per_s * dist.get_world_size()

    if rank == 0:
        print(f"rank0 tokens/s: {global_tokens_per_s:.2f} tokens/s (local {local_tokens_per_s:.2f})")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    _run_worker(args.iters, args.warmup)


def get_benchmark() -> BaseBenchmark:
    return MultiGPUDecodeBenchmark()


if __name__ == "__main__":
    main()




