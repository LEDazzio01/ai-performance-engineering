#!/usr/bin/env python3

import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

"""Branch-free ReLU-style threshold operation example referenced in Chapter 8."""

import torch


def threshold_op(x: torch.Tensor) -> torch.Tensor:
    """Compute max(x, 0) using vectorized tensor ops to avoid warp divergence."""
    zero = torch.zeros_like(x)
    return torch.maximum(x, zero)


def main() -> None:
    torch.manual_seed(0)
    n = 1_000_000
    x = torch.randn(n, device="cuda")
    y = threshold_op(x)
    torch.cuda.synchronize()
    print(f"Computed threshold_op on {y.numel()} elements; sample mean={y.mean().item():.4f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for threshold_op demo.")
    main()

