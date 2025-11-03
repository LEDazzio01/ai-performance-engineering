#!/usr/bin/env python3
"""torch.compile version of the Chapter 8 threshold example."""

import torch


@torch.compile
def threshold_op(x: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(x)
    return torch.maximum(x, zero)


def main() -> None:
    torch.manual_seed(0)
    n = 1_000_000
    x = torch.randn(n, device="cuda")
    y = threshold_op(x)
    torch.cuda.synchronize()
    print(f"Compiled threshold_op complete; sample mean={y.mean().item():.4f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for jit_threshold_op demo.")
    main()

