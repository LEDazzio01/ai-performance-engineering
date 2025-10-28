"""PyTorch memory access patterns benchmark."""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import time
import torch

N = 1 << 20

def benchmark_copy(style: str) -> float:
    src = torch.arange(N, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    
    # Use CUDA Events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    if style == "scalar":
        dst.copy_(src)
    elif style == "vectorized":
        dst.copy_(src)
    end_event.record()
    end_event.synchronize()
    
    return start_event.elapsed_time(end_event)  # Already in ms


def main() -> None:
    torch.cuda.init()
    ms = benchmark_copy("scalar")
    print(f"scalar copy: {ms:.2f} ms")
    ms = benchmark_copy("vectorized")
    print(f"vectorized copy: {ms:.2f} ms")


if __name__ == "__main__":
    main()
