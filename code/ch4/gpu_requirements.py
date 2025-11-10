import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

"""Helper module for checking GPU requirements before running multi-GPU scripts."""

import torch


def require_min_gpus(min_gpus: int, script_name: str = None) -> None:
    """
    Check if system has minimum required GPUs, fail fast with clear message if not.
    
    Args:
        min_gpus: Minimum number of GPUs required
        script_name: Name of the script (for error message)
    """
    available_gpus = torch.cuda.device_count()
    
    if available_gpus < min_gpus:
        script = script_name or sys.argv[0]
        print(f"╔{'═' * 78}╗", file=sys.stderr)
        print(f"║ {'GPU REQUIREMENT NOT MET':^76} ║", file=sys.stderr)
        print(f"╠{'═' * 78}╣", file=sys.stderr)
        print(f"║ Script: {script:<69} ║", file=sys.stderr)
        print(f"║ Required GPUs: {min_gpus:<62} ║", file=sys.stderr)
        print(f"║ Available GPUs: {available_gpus:<61} ║", file=sys.stderr)
        print(f"║{' ' * 78}║", file=sys.stderr)
        print(f"║ This script requires at least {min_gpus} GPU(s) to run correctly.{' ' * (35 - len(str(min_gpus)))} ║", file=sys.stderr)
        print(f"║ Current system has {available_gpus} GPU(s) available.{' ' * (41 - len(str(available_gpus)))} ║", file=sys.stderr)
        print(f"║{' ' * 78}║", file=sys.stderr)
        print(f"║ To run this script:{' ' * 58}║", file=sys.stderr)
        print(f"║ • Use a system with {min_gpus}+ GPUs{' ' * (54 - len(str(min_gpus)))} ║", file=sys.stderr)
        print(f"║ • Or modify the script to work with fewer GPUs{' ' * 30} ║", file=sys.stderr)
        print(f"╚{'═' * 78}╝", file=sys.stderr)
        sys.exit(1)


def skip_if_insufficient_gpus(min_gpus: int = 2) -> None:
    """
    Raise a standardized SKIPPED RuntimeError when the system lacks enough GPUs.
    
    This lets the benchmark harness record hardware limitations instead of
    counting the benchmark as a failure in smoke-test mode.
    """
    available_gpus = torch.cuda.device_count()
    if available_gpus < min_gpus:
        raise RuntimeError(
            f"SKIPPED: Distributed benchmark requires multiple GPUs (found {available_gpus} GPU)"
        )


def warn_optimal_gpu_count(optimal_gpus: int, script_name: str = None) -> None:
    """
    Warn if system doesn't have optimal GPU count but can still run.
    
    Args:
        optimal_gpus: Optimal number of GPUs
        script_name: Name of the script (for warning message)
    """
    available_gpus = torch.cuda.device_count()
    
    if available_gpus < optimal_gpus:
        script = script_name or sys.argv[0]
        print(f"\nWARNING: WARNING: {script}", file=sys.stderr)
        print(f"    This script is optimized for {optimal_gpus} GPUs but only {available_gpus} available.", file=sys.stderr)
        print(f"    Performance may be suboptimal.\n", file=sys.stderr)
