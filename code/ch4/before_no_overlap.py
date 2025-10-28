"""Baseline DDP example without communication/computation overlap.

Updated to reflect Chapter 4 best practices:
- Uses env:// initialization with MASTER_ADDR/PORT.
- Avoids passing large tensors through mp.spawn; each rank synthesizes its own data.
- Guards multiprocessing start method.
- Cleans up the process group on exit.
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import os
import socket
import sys
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


def init_process(rank: int, world_size: int) -> None:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29501"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


class MultiLayerNet(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def synthetic_batch(feature_dim: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    data = torch.randn(batch_size, feature_dim, generator=gen)
    target = torch.randn(batch_size, 1, generator=gen)
    return data, target


def train_no_overlap(rank: int, world_size: int, feature_dim: int = 1024, batch_size: int = 128) -> None:
    if torch.cuda.device_count() < world_size:
        print(f"[Rank {rank}] Insufficient GPUs ({torch.cuda.device_count()}) for world_size={world_size}; running single process.", flush=True)
        world_size = 1

    init_process(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = MultiLayerNet(feature_dim).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device] if device.type == "cuda" else None)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data, target = synthetic_batch(feature_dim, batch_size)
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()

    if world_size > 1:
        for param in model.parameters():  # Manual gradient reduction (no overlap)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

    optimizer.step()

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        print(f"Loss: {loss.item():.4f}", flush=True)


def main() -> None:
    desired_world = min(2, torch.cuda.device_count() or 1)
    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)
        if desired_world > 1:
            mp.spawn(train_no_overlap, args=(desired_world,), nprocs=desired_world, join=True)
        else:
            train_no_overlap(0, 1)


if __name__ == "__main__":
    main()
