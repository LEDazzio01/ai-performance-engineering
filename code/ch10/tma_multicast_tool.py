#!/usr/bin/env python3
"""TMA multicast demo runner (chapter tool).

This is intentionally NOT a comparative benchmark pair for `aisp bench run`.
Run via:
  python -m cli.aisp tools tma-multicast -- --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _detect_sm_suffix() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required to detect GPU arch for this tool") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this tool")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return f"_sm{sm}"


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Chapter 10 TMA multicast demo binaries.")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building; assumes the binaries are already built.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline binary.",
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Run only the cluster/multicast binary.",
    )
    args = parser.parse_args(argv)

    chapter_dir = Path(__file__).resolve().parent
    suffix = _detect_sm_suffix()
    baseline = f"tma_multicast_baseline{suffix}"
    cluster = f"tma_multicast_cluster{suffix}"

    run_baseline = args.baseline_only or not args.cluster_only
    run_cluster = args.cluster_only or not args.baseline_only

    if not args.no_build:
        targets = []
        if run_baseline:
            targets.append(baseline)
        if run_cluster:
            targets.append(cluster)
        _run(["make", "-j", "4", *targets], cwd=chapter_dir)

    if run_baseline:
        print(f"\n=== {baseline} ===")
        _run([f"./{baseline}"], cwd=chapter_dir)
    if run_cluster:
        print(f"\n=== {cluster} ===")
        _run([f"./{cluster}"], cwd=chapter_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

