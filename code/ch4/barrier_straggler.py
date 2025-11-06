from __future__ import annotations
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

"""Demonstrate how monitored_barrier surfaces stragglers."""


import datetime
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

try:
    from gpu_requirements import require_min_gpus, warn_optimal_gpu_count
except ImportError:
    def require_min_gpus(min_gpus, script_name=None):
        import torch, sys
        if torch.cuda.device_count() < min_gpus:
            print(f"ERROR: This script requires {min_gpus} GPUs but only {torch.cuda.device_count()} available", file=sys.stderr)
            sys.exit(1)
    def warn_optimal_gpu_count(optimal_gpus, script_name=None):
        pass


def init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA devices required for straggler demo.")

    if not dist.is_initialized():
        setup_single_gpu_env()  # Auto-setup for single-GPU mode
    # Use GLOO backend since monitored_barrier is only implemented for GLOO
    dist.init_process_group("gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def main() -> None:
    # Check GPU requirements early - barrier testing needs multiple GPUs
    require_min_gpus(2, "barrier_straggler.py")
    
    rank, world_size, device = init_distributed()

    if rank == 1 and world_size > 1:
        time.sleep(2.0)  # Inject artificial delay

    dummy_tensor = torch.randn(1000, 1000, device=device)
    _ = torch.mm(dummy_tensor, dummy_tensor.t())

    try:
        start = time.time()
        dist.monitored_barrier(timeout=datetime.timedelta(seconds=30))
        elapsed = time.time() - start
        print(f"Rank {rank} completed barrier in {elapsed:.2f}s", flush=True)
    except RuntimeError as exc:
        print(f"Rank {rank} barrier timeout: {exc}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
