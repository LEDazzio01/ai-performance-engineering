#!/usr/bin/env python3
"""torch.compile version of the Chapter 8 threshold example.

Chapter 8: Double Buffering and Pipelining

This file demonstrates using torch.compile for kernel optimization.

FORWARD REFERENCE: torch.compile is covered in depth in Chapter 14 
(TorchInductor). Here we use it to show how the compiler can optimize
simple threshold operations. See ch14/*compile*.py for detailed analysis.
"""

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

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

