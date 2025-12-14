#!/usr/bin/env python3
"""TMEM triple-overlap demo runner (chapter tool).

Wraps the CUDA 13 Blackwell TMA 2D pipeline sample. This is intentionally NOT a
comparative benchmark pair for `aisp bench run`.

Run via:
  python -m cli.aisp tools tmem-triple-overlap -- --help
"""

from __future__ import annotations

import argparse
import subprocess
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
    parser = argparse.ArgumentParser(description="Run the Chapter 10 TMEM triple-overlap demo binary.")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building; assumes the binary is already built.",
    )
    parser.add_argument(
        "tool_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the demo binary (prefix with --).",
    )
    args = parser.parse_args(argv)

    chapter_dir = Path(__file__).resolve().parent
    suffix = _detect_sm_suffix()
    binary = f"tma_2d_pipeline_blackwell{suffix}"

    if not args.no_build:
        _run(["make", "-j", "4", binary], cwd=chapter_dir)

    cmd = [f"./{binary}", *args.tool_args]
    print(f"\n=== {binary} ===")
    _run(cmd, cwd=chapter_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

