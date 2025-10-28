"""Naive sequential PyTorch loop (illustrates GPU under-utilization)."""
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import time
import torch

N = 10_000

def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example")

    device = torch.device("cuda")
    A = torch.arange(N, dtype=torch.float32, device=device)
    B = 2 * A
    C = torch.empty_like(A)

    torch.cuda.synchronize()
    start = time.time()

    for i in range(N):  # intentionally launches N tiny kernels
        C[i] = A[i] + B[i]

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1_000
    print(f"Sequential PyTorch loop took {elapsed_ms:.2f} ms")
    print(f"Result: C[0]={C[0].item():.1f}, C[-1]={C[-1].item():.1f}")

if __name__ == "__main__":
    main()
