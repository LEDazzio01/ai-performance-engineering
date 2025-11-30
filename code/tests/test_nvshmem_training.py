import os
import socket
import sys
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _setup_env(world_size: int, port: int, rank: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)


def _gradient_worker(rank: int, world_size: int, seq_len: int, port: int) -> None:
    _setup_env(world_size, port, rank)
    from ch04.nvshmem_training_example import (
        TransformerBlock,
        demo_gradient_buckets,
    )

    torch.manual_seed(0)
    batch = torch.randn(2, seq_len, 16)
    model = TransformerBlock(d_model=16, n_heads=4, mlp_ratio=2)
    demo_gradient_buckets(batch, model)
    if dist.is_initialized():
        dist.destroy_process_group()


def _pipeline_worker(rank: int, world_size: int, seq_len: int, port: int) -> None:
    _setup_env(world_size, port, rank)
    from ch04.nvshmem_training_example import demo_pipeline_parallel

    torch.manual_seed(0)
    microbatch = torch.randn(2, seq_len, 16)
    demo_pipeline_parallel(microbatch)
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_gradient_bucket_demo_runs_on_gloo():
    from ch04.nvshmem_training_example import nvshmem_available

    if not nvshmem_available() or torch.cuda.device_count() < 2:
        return

    world_size = 2
    port = _find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    spawn(_gradient_worker, args=(world_size, 8, port), nprocs=world_size, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_pipeline_demo_runs_on_gloo():
    from ch04.nvshmem_training_example import nvshmem_available

    if not nvshmem_available() or torch.cuda.device_count() < 2:
        return

    world_size = 2
    port = _find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    spawn(_pipeline_worker, args=(world_size, 8, port), nprocs=world_size, join=True)
