#!/usr/bin/env python3
"""
TorchTitan Async Tensor Parallelism demo
----------------------------------------

This example shows how to enable the experimental Async Tensor Parallelism
feature that ships with TorchTitan (https://github.com/pytorch/torchtitan).
Run it with torchrun on a node with multiple CUDA devices, e.g.:

    torchrun --nproc_per_node=2 ch4/torchtitan_async_tp_demo.py --tp-degree 2

The script:
  * creates a simple MLP and shards it with torch.distributed.tensor.parallel,
  * enables Async-TP via torchtitan.distributed.tensor_parallel.maybe_enable_async_tp,
  * compiles the model with torch.compile so micro-pipelining can overlap
    communication with compute, and
  * runs a few warm-up/training steps while logging overlap-friendly timings.

Async-TP currently requires:
  * CUDA build of PyTorch 2.9+ (nightly or stable with CUDA 13),
  * TorchTitan >= 0.2.0 (pip install torchtitan),
  * NCCL backend (infers symmetric memory collectives automatically),
  * torch.compile mode that includes the model ("max-autotune" by default below).
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    Replicate,
    RowwiseParallel,
    Shard,
    parallelize_module,
)

try:
    from torchtitan.config.job_config import JobConfig
    from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
except ImportError as err:  # pragma: no cover - informative error path
    raise SystemExit(
        "torchTitan is required for this example. Install with `pip install torchtitan`."
    ) from err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchTitan Async-TP demo")
    parser.add_argument("--tp-degree", type=int, default=2, help="Tensor parallel degree / world size")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden size for the toy MLP")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank micro-batch size")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps to run")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices={"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"},
        help="torch.compile mode used for the demo",
    )
    return parser.parse_args()


def init_distributed(tp_degree: int) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA devices are required to run this demo.")

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:  # pragma: no cover - torchrun always sets LOCAL_RANK
        local_rank = 0

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    if world_size != tp_degree:
        raise SystemExit(
            f"World size ({world_size}) must match --tp-degree ({tp_degree}) for this demo."
        )

    torch.cuda.set_device(local_rank)
    return local_rank


class ToyMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.act = torch.nn.GELU(approximate="tanh")
        self.fc2 = torch.nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def enable_async_tp(tp_mesh, compile_mode: str) -> JobConfig:
    job_cfg = JobConfig()
    job_cfg.parallelism.tensor_parallel_degree = tp_mesh.size()
    job_cfg.parallelism.enable_async_tensor_parallel = True
    job_cfg.compile.enable = True
    job_cfg.compile.components = ["model"]
    job_cfg.compile.backend = "inductor"
    
    # PyTorch 2.9: Enhanced compilation options
    job_cfg.compile.mode = compile_mode
    if hasattr(job_cfg.compile, "inductor_config"):
        # Enable CUDA graph trees (PyTorch 2.9)
        job_cfg.compile.inductor_config.triton_cudagraphs = True
        job_cfg.compile.inductor_config.triton_cudagraph_trees = True

    maybe_enable_async_tp(job_cfg, tp_mesh)
    return job_cfg


def shard_model(model: torch.nn.Module, tp_mesh) -> None:
    parallelize_module(
        model,
        tp_mesh,
        parallelize_plan={
            "fc1": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "fc2": RowwiseParallel(output_layouts=Replicate()),
            "act": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
            ),
        },
    )


def main() -> None:
    args = parse_args()
    local_rank = init_distributed(args.tp_degree)
    device = torch.device("cuda", local_rank)

    tp_mesh = init_device_mesh("cuda", (args.tp_degree,), mesh_dim_names=("tp",))
    enable_async_tp(tp_mesh, args.compile_mode)

    model = ToyMLP(hidden_dim=args.hidden_dim).to(device)
    shard_model(model, tp_mesh)
    compiled_model = torch.compile(model, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=3e-4, fused=True)
    loss_fn = torch.nn.MSELoss()

    x = torch.randn(args.batch_size, args.hidden_dim, device=device)
    target = torch.zeros_like(x)

    torch.cuda.synchronize()
    dist.barrier()

    for step in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        out = compiled_model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if dist.get_rank() == 0:
            print(
                f"[step {step:02d}] loss={loss.item():.5f} "
                f"(Async-TP micro-pipelined step took {elapsed_ms:.2f} ms)"
            )

    dist.destroy_process_group()


if __name__ == "__main__":  # pragma: no cover - torchrun entry point
    main()
