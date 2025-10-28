"""Dynamic allocator retry helper for Chapter 19."""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def allocate_on_gpu(size_mb: int) -> None:
    tensors = []
    try:
        for _ in range(8):
            tensors.append(torch.randn(size_mb * 256, 1024, device="cuda"))
            time.sleep(0.05)
    except RuntimeError as exc:
        print(f"Allocation failed: {exc}")
        raise
    finally:
        del tensors
        torch.cuda.empty_cache()
        gc.collect()


def run_child(size_mb: int) -> None:
    torch.cuda.set_device(0)
    allocate_on_gpu(size_mb)


def run_parent(size_mb: int) -> None:
    script = Path(__file__).resolve()
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        subprocess.run([sys.executable, str(script), "--child", f"--size-mb={size_mb}"], check=True, env=env)
    except subprocess.CalledProcessError:
        print("Retrying after cleanup with half the allocation size...")
        torch.cuda.empty_cache()
        gc.collect()
        new_size = max(32, size_mb // 2)
        subprocess.run([sys.executable, str(script), "--child", f"--size-mb={new_size}"], check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Allocator retry helper")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--size-mb", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.child:
        run_child(args.size_mb)
    else:
        run_parent(args.size_mb)


if __name__ == "__main__":
    main()
