#!/usr/bin/env python3
"""Optimized: Tensor Parallelism with communication overlap.

Launches async all-gather on a dedicated stream and overlaps with compute.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add common to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.distributed as dist

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BATCH = 8
_DEFAULT_SEQ = 2048
_DEFAULT_HIDDEN = 4096
_DEFAULT_LAYERS = 4


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("optimized_tensor_parallel_async requires torchrun (RANK/WORLD_SIZE missing).")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _build_layers(hidden: int, hidden_per_rank: int, num_layers: int, device: torch.device):
    shard = nn.ModuleList([
        nn.Linear(hidden, hidden_per_rank, bias=False)
        for _ in range(num_layers)
    ]).to(device).to(torch.bfloat16)
    proj = nn.ModuleList([
        nn.Linear(hidden, hidden, bias=False)
        for _ in range(num_layers)
    ]).to(device).to(torch.bfloat16)
    aux = nn.ModuleList([
        nn.Linear(hidden, hidden, bias=False)
        for _ in range(num_layers)
    ]).to(device).to(torch.bfloat16)
    return shard, proj, aux


def _run_worker(
    iters: int,
    warmup: int,
    batch: int,
    seq_length: int,
    hidden: int,
    num_layers: int,
) -> None:
    rank, world_size, local_rank = _init_distributed()
    if world_size < 2:
        raise RuntimeError("optimized_tensor_parallel_async requires >=2 GPUs.")
    if hidden % world_size != 0:
        raise ValueError("hidden_size must be divisible by world_size")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    hidden_per_rank = hidden // world_size

    shard_layers, proj_layers, aux_layers = _build_layers(hidden, hidden_per_rank, num_layers, device)
    inputs = torch.randn(batch, seq_length, hidden, device=device, dtype=torch.bfloat16)
    gather_list = [torch.empty(batch, seq_length, hidden_per_rank, device=device, dtype=torch.bfloat16)
                   for _ in range(world_size)]
    comm_stream = torch.cuda.Stream()

    def _step() -> None:
        x = inputs
        for layer_idx in range(num_layers):
            local_out = shard_layers[layer_idx](x)
            comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm_stream):
                work = dist.all_gather(gather_list, local_out, async_op=True)
            aux_out = aux_layers[layer_idx](x)
            work.wait()
            full_out = torch.cat(gather_list, dim=-1)
            proj_out = proj_layers[layer_idx](full_out)
            x = proj_out + aux_out

    for _ in range(max(warmup, 0)):
        _step()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        _step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens_per_iter = batch * seq_length
    tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
    if rank == 0:
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized tensor parallel benchmark")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--seq-length", type=int, default=_DEFAULT_SEQ)
    parser.add_argument("--hidden-size", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument("--num-layers", type=int, default=_DEFAULT_LAYERS)
    args = parser.parse_args()
    _run_worker(
        args.iters,
        args.warmup,
        args.batch_size,
        args.seq_length,
        args.hidden_size,
        args.num_layers,
    )


class OptimizedTensorParallelBenchmark(BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    verification_not_applicable_reason = "torchrun benchmarks execute in external processes"
    skip_input_check = True
    skip_output_check = True

    def __init__(self) -> None:
        super().__init__()
        tokens = float(_DEFAULT_BATCH * _DEFAULT_SEQ)
        self.register_workload_metadata(requests_per_iteration=float(_DEFAULT_BATCH), tokens_per_iteration=tokens)

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: optimized_tensor_parallel_async requires torchrun")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=2,
            iterations=3,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="optimized_tensor_parallel_async",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )

    def get_input_signature(self) -> dict:
        signature = simple_signature(
            batch_size=_DEFAULT_BATCH,
            dtype="bfloat16",
            seq_length=_DEFAULT_SEQ,
            hidden_size=_DEFAULT_HIDDEN,
            num_layers=_DEFAULT_LAYERS,
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
        )
        signature.world_size = 2
        signature.collective_type = "all_gather"
        return signature

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedTensorParallelBenchmark()


if __name__ == "__main__":
    main()
