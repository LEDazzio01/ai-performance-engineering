"""GPU requirement helpers for Chapter 5 benchmarks."""

from __future__ import annotations

import sys

import torch


def skip_if_insufficient_gpus(min_gpus: int = 2) -> None:
    """
    Raise a standardized SKIPPED RuntimeError when not enough GPUs are present.

    Benchmarks that truly require multiple GPUs should call this during setup so
    the harness records a hardware limitation instead of a failure.
    """
    available_gpus = torch.cuda.device_count()
    if available_gpus < min_gpus:
        raise RuntimeError(
            f"SKIPPED: Distributed benchmark requires multiple GPUs (found {available_gpus} GPU)"
        )


def require_min_gpus(min_gpus: int, script_name: str | None = None) -> None:
    """
    Legacy helper for standalone scripts (mirrors the Chapter 4 version).

    Prints a descriptive error and exits if the system does not meet the GPU
    requirement. Prefer skip_if_insufficient_gpus() when running under the
    benchmark harness.
    """
    available_gpus = torch.cuda.device_count()
    if available_gpus >= min_gpus:
        return

    script = script_name or sys.argv[0]
    message = [
        "╔" + "═" * 78 + "╗",
        f"║ {'GPU REQUIREMENT NOT MET':^76} ║",
        "╠" + "═" * 78 + "╣",
        f"║ Script: {script:<69} ║",
        f"║ Required GPUs: {min_gpus:<62} ║",
        f"║ Available GPUs: {available_gpus:<61} ║",
        "║" + " " * 78 + "║",
        f"║ This script requires at least {min_gpus} GPU(s) to run correctly."
        + " " * (35 - len(str(min_gpus))) + "║",
        f"║ Current system has {available_gpus} GPU(s) available."
        + " " * (41 - len(str(available_gpus))) + "║",
        "║" + " " * 78 + "║",
        "║ To run this script:" + " " * 58 + "║",
        f"║ • Use a system with {min_gpus}+ GPUs"
        + " " * (54 - len(str(min_gpus))) + "║",
        "║ • Or modify the script to work with fewer GPUs"
        + " " * 30 + "║",
        "╚" + "═" * 78 + "╝",
    ]
    for line in message:
        print(line, file=sys.stderr)
    sys.exit(1)
