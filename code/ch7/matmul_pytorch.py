from __future__ import annotations
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

"""PyTorch naive vs vectorized matmul benchmark."""

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import time
import torch

M = 512
N = 512
K = 512


def benchmark(op) -> float:
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    
    # Use CUDA Events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    op(a, b)
    end_event.record()
    end_event.synchronize()
    
    return start_event.elapsed_time(end_event)  # Already in ms


def main() -> None:
    torch.cuda.init()
    naive_time = benchmark(lambda x, y: torch.einsum("ik,kj->ij", x, y))
    optimized_time = benchmark(lambda x, y: torch.matmul(x, y))
    print(f"naive einsum: {naive_time:.2f} ms")
    print(f"torch.matmul: {optimized_time:.2f} ms")


if __name__ == "__main__":
    main()
