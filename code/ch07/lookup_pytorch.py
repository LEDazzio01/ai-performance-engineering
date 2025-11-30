from __future__ import annotations
"""GPU gather benchmark comparing naive vs. coalesced access.

This replicates the Chapter 7 guidance with simple timing.
"""
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path


import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import time
import torch

N = 1 << 20


def run(indices: torch.Tensor) -> float:
    table = torch.arange(N, device=indices.device, dtype=torch.float32)
    out = torch.empty_like(table)
    
    if indices.device.type == "cuda":
        # Use CUDA Events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        out = table[indices]
        end_event.record()
        end_event.synchronize()
        
        return float(start_event.elapsed_time(end_event))  # Already in ms
    else:
        # CPU timing
        import time
        start = time.time()
        out = table[indices]
        return (time.time() - start) * 1_000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = torch.randint(0, N, (N,), device=device)
    ms = run(indices)
    print(f"random gather: {ms:.2f} ms")
    indices = torch.arange(N, device=device)
    ms = run(indices)
    print(f"coalesced gather: {ms:.2f} ms")


if __name__ == "__main__":
    main()
