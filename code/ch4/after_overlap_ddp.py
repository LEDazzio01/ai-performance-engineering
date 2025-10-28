"""torch.distributed example illustrating DDP's communication/computation overlap.

This version aligns with Chapter 4 guidance:
- Environment-based init (MASTER_ADDR/PORT).
- Each rank generates its own synthetic data (no large tensors passed through spawn).
- Multiprocessing start method guarded.
- Clean group teardown.
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


def init_process(rank: int, world_size: int) -> None:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29502"
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


def synthetic_batch(feature_dim: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    data = torch.randn(batch_size, feature_dim, generator=gen)
    target = torch.randn(batch_size, 1, generator=gen)
    return data, target


def train_ddp(rank: int, world_size: int, feature_dim: int = 1024, batch_size: int = 128) -> None:
    init_process(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = MultiLayerNet(feature_dim).to(device)

    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device] if device.type == "cuda" else None,
        gradient_as_bucket_view=True,  # larger buckets help overlap
    )

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    data, target = synthetic_batch(feature_dim, batch_size)
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()  # DDP schedules gradient all-reduce in the background
    optimizer.step()

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        print(f"DDP loss: {loss.item():.4f}", flush=True)


def main() -> None:
    world_size = min(2, torch.cuda.device_count() or 1)
    mp.set_start_method("spawn", force=True)

    if world_size > 1:
        mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Only one GPU present; running DDP demo with world_size=1")
        train_ddp(0, 1)


if __name__ == "__main__":
    main()
