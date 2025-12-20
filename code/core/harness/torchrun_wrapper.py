"""Torchrun wrapper for benchmark launches.

This wrapper exists to enforce harness-level invariants inside torchrun-launched
multi-process benchmarks.

Currently enforced:
- RNG seed immutability: benchmarks must not reseed away from the harness-
  configured seeds (default seed=42).

The harness launches torchrun with this wrapper as the entrypoint and passes the
original benchmark script path + args through unchanged.
"""

from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.harness.backend_policy import BackendPolicyName, apply_backend_policy


def _apply_backend_policy(deterministic: bool) -> None:
    apply_backend_policy(BackendPolicyName.PERFORMANCE, deterministic)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_target_script(script_path: Path, argv: list[str]) -> None:
    previous_argv = sys.argv
    previous_path0: Optional[str] = None
    try:
        sys.argv = [str(script_path), *argv]
        if sys.path:
            previous_path0 = sys.path[0]
            sys.path[0] = str(script_path.parent)
        else:
            sys.path.insert(0, str(script_path.parent))
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = previous_argv
        if previous_path0 is None:
            if sys.path and sys.path[0] == str(script_path.parent):
                sys.path.pop(0)
        else:
            if sys.path:
                sys.path[0] = previous_path0


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--aisp-target-script",
        required=True,
        help="Path to the benchmark script to execute under torchrun.",
    )
    parser.add_argument(
        "--aisp-expected-torch-seed",
        required=True,
        type=int,
        help="Expected torch.initial_seed() after benchmark completes.",
    )
    parser.add_argument(
        "--aisp-expected-cuda-seed",
        required=False,
        type=int,
        help="Expected torch.cuda.initial_seed() after benchmark completes (if CUDA is available).",
    )
    parser.add_argument(
        "--aisp-deterministic",
        action="store_true",
        help="Enable deterministic algorithms (mirrors harness deterministic mode).",
    )
    args, remainder = parser.parse_known_args(argv)

    script_path = Path(args.aisp_target_script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Target script not found: {script_path}")

    _apply_backend_policy(bool(args.aisp_deterministic))
    _set_seeds(int(args.aisp_expected_torch_seed))

    expected_torch_seed = int(args.aisp_expected_torch_seed)
    expected_cuda_seed: Optional[int] = args.aisp_expected_cuda_seed

    _run_target_script(script_path, remainder)

    current_torch_seed = int(torch.initial_seed())
    if current_torch_seed != expected_torch_seed:
        raise RuntimeError(
            "Seed mutation detected during torchrun benchmark execution. "
            f"Expected torch.initial_seed()={expected_torch_seed}, got {current_torch_seed}. "
            "Benchmarks MUST NOT reseed; rely on harness-configured seeds."
        )

    if torch.cuda.is_available():
        if expected_cuda_seed is None:
            raise RuntimeError(
                "torch.cuda.is_available() is true but --aisp-expected-cuda-seed was not provided."
            )
        current_cuda_seed = int(torch.cuda.initial_seed())
        if current_cuda_seed != int(expected_cuda_seed):
            raise RuntimeError(
                "CUDA seed mutation detected during torchrun benchmark execution. "
                f"Expected torch.cuda.initial_seed()={int(expected_cuda_seed)}, got {current_cuda_seed}. "
                "Benchmarks MUST NOT reseed; rely on harness-configured seeds."
            )


if __name__ == "__main__":
    main()
