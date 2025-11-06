from __future__ import annotations
"""Simulate UCX/NCCL memory registration patterns to highlight fragmentation risks.

Launch:
    torchrun --standalone --nproc-per-node=2 extras/ch4/ucx_fragmentation.py
"""
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path



import os
import time

import torch
import torch.distributed as dist

try:
    from distributed_helper import setup_single_gpu_env
except ImportError:
    def setup_single_gpu_env():
        if "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("LOCAL_RANK", "0")


def log_mem(iteration: int, device: torch.device) -> None:
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    print(
        f"[Iter {iteration:02d}] Reserved: {reserved / 1e9:.3f} GB, "
        f"Allocated: {allocated / 1e9:.3f} GB",
        flush=True,
    )


def init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for UCX fragmentation demo.")

    if not dist.is_initialized():
        setup_single_gpu_env()  # Auto-setup for single-GPU mode
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def simulate(rank: int, world_size: int, device: torch.device) -> None:
    # Ensure a large persistent registration (simulated UCX pinned buffer)
    big_buffer = torch.empty(int(2e8), device=device)  # ~0.8 GB
    log_mem(0, device)

    for i in range(10):
        # Variable-size allocations trigger UCX memory registrations
        small_tensor = torch.randn(1000 + i * 100, 1000, device=device)

        if world_size > 1:
            dist.all_reduce(small_tensor)
        else:
            _ = small_tensor * 2

        if i % 3 == 0:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        log_mem(i + 1, device)
        del small_tensor
        time.sleep(0.05)

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        print("UCX fragmentation simulation complete.", flush=True)


def main() -> None:
    # Favor UCX/RDMA paths when available
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "3")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")

    rank, world_size, device = init_distributed()
    simulate(rank, world_size, device)


if __name__ == "__main__":
    main()
