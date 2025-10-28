"""Vectorized PyTorch addition (correct GPU utilization)."""
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import time
import torch

N = 1_000_000

def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example")

    device = torch.device("cuda")
    A = torch.arange(N, dtype=torch.float32, device=device)
    B = 2 * A

    torch.cuda.synchronize()
    start = time.time()
    C = A + B
    torch.cuda.synchronize()

    elapsed_ms = (time.time() - start) * 1_000
    print(f"Vectorized PyTorch add took {elapsed_ms:.2f} ms")
    print(f"Result: C[0]={C[0].item():.1f}, C[-1]={C[-1].item():.1f}")

if __name__ == "__main__":
    main()
