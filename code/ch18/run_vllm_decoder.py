#!/usr/bin/env python3
"""Simple wrapper that invokes the FlexDecoding example used in the book."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    script = Path(__file__).with_name("flexdecoding_example.py")
    cmd = [
        sys.executable,
        str(script),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
