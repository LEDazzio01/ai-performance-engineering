#!/usr/bin/env python3
"""Optimized: Pipeline Parallelism (1F1B schedule).

Interleaves forward and backward micro-batches to reduce pipeline bubbles.
Launched via torchrun.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
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

_DEFAULT_BATCH = 32
_DEFAULT_SEQ = 2048
_DEFAULT_HIDDEN = 4096
_DEFAULT_LAYERS = 8
_DEFAULT_MICRO_BATCHES = 8


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("optimized_pipeline_parallel_1f1b requires torchrun (RANK/WORLD_SIZE missing).")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _build_stage_layers(hidden: int, layers_per_stage: int, device: torch.device):
    fwd = nn.ModuleList([
        nn.Linear(hidden, hidden, bias=False)
        for _ in range(layers_per_stage)
    ]).to(device).to(torch.bfloat16)
    bwd = nn.ModuleList([
        nn.Linear(hidden, hidden, bias=False)
        for _ in range(layers_per_stage)
    ]).to(device).to(torch.bfloat16)
    return fwd, bwd


def _run_worker(
    iters: int,
    warmup: int,
    batch_size: int,
    seq_length: int,
    hidden: int,
    num_layers: int,
    num_micro_batches: int,
) -> None:
    rank, world_size, local_rank = _init_distributed()
    if world_size < 2:
        raise RuntimeError("optimized_pipeline_parallel_1f1b requires >=2 GPUs.")
    if num_layers % world_size != 0:
        raise ValueError("num_layers must be divisible by world_size")
    if batch_size % num_micro_batches != 0:
        raise ValueError("batch_size must be divisible by num_micro_batches")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    layers_per_stage = num_layers // world_size
    micro_batch_size = batch_size // num_micro_batches

    fwd_layers, bwd_layers = _build_stage_layers(hidden, layers_per_stage, device)

    if rank == 0:
        inputs = torch.randn(batch_size, seq_length, hidden, device=device, dtype=torch.bfloat16)
    else:
        inputs = None

    def _forward(micro_batch: torch.Tensor) -> torch.Tensor:
        x = micro_batch
        for layer in fwd_layers:
            x = torch.relu(layer(x))
        return x

    def _backward(grad_in: torch.Tensor) -> torch.Tensor:
        x = grad_in
        for layer in bwd_layers:
            x = torch.relu(layer(x))
        return x

    def _run_iteration() -> None:
        activations: deque[torch.Tensor] = deque()
        warmup_steps = min(world_size, num_micro_batches)

        for micro_idx in range(warmup_steps):
            if rank == 0:
                start_idx = micro_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                micro_batch = inputs[start_idx:end_idx]
            else:
                micro_batch = torch.empty(
                    micro_batch_size,
                    seq_length,
                    hidden,
                    device=device,
                    dtype=torch.bfloat16,
                )
                dist.recv(micro_batch, src=rank - 1)

            out = _forward(micro_batch)
            activations.append(out)
            if rank < world_size - 1:
                dist.send(out, dst=rank + 1)

        for micro_idx in range(warmup_steps, num_micro_batches):
            activation = activations.popleft()
            if rank < world_size - 1:
                grad_in = torch.empty_like(activation)
                dist.recv(grad_in, src=rank + 1)
            else:
                grad_in = activation

            grad = _backward(grad_in)
            if rank > 0:
                dist.send(grad, dst=rank - 1)

            if rank == 0:
                start_idx = micro_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                micro_batch = inputs[start_idx:end_idx]
            else:
                micro_batch = torch.empty(
                    micro_batch_size,
                    seq_length,
                    hidden,
                    device=device,
                    dtype=torch.bfloat16,
                )
                dist.recv(micro_batch, src=rank - 1)

            out = _forward(micro_batch)
            activations.append(out)
            if rank < world_size - 1:
                dist.send(out, dst=rank + 1)

        while activations:
            activation = activations.popleft()
            if rank < world_size - 1:
                grad_in = torch.empty_like(activation)
                dist.recv(grad_in, src=rank + 1)
            else:
                grad_in = activation

            grad = _backward(grad_in)
            if rank > 0:
                dist.send(grad, dst=rank - 1)

    for _ in range(max(warmup, 0)):
        _run_iteration()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        _run_iteration()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if rank == 0:
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized 1F1B pipeline parallel benchmark")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--seq-length", type=int, default=_DEFAULT_SEQ)
    parser.add_argument("--hidden-size", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument("--num-layers", type=int, default=_DEFAULT_LAYERS)
    parser.add_argument("--micro-batches", type=int, default=_DEFAULT_MICRO_BATCHES)
    args = parser.parse_args()
    _run_worker(
        args.iters,
        args.warmup,
        args.batch_size,
        args.seq_length,
        args.hidden_size,
        args.num_layers,
        args.micro_batches,
    )


class OptimizedPipelineParallelBenchmark(BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    verification_not_applicable_reason = "torchrun benchmarks execute in external processes"
    skip_input_check = True
    skip_output_check = True

    def __init__(self) -> None:
        super().__init__()
        tokens = float(_DEFAULT_BATCH * _DEFAULT_SEQ)
        self.register_workload_metadata(requests_per_iteration=float(_DEFAULT_BATCH), tokens_per_iteration=tokens)

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: optimized_pipeline_parallel_1f1b requires torchrun")

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
            name="optimized_pipeline_parallel_1f1b",
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
        signature.pipeline_stages = 2
        signature.collective_type = "send_recv"
        return signature

    def get_output_tolerance(self) -> tuple:
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedPipelineParallelBenchmark()


if __name__ == "__main__":
    main()
