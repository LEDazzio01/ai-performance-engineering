"""Multi-GPU wrapper for symmetric memory examples; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch04.symmetric_memory_example import main as symmetric_memory_main


def get_benchmark():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: symmetric_memory_* requires >=2 GPUs")

    def _run():
        symmetric_memory_main()

    return _run


if __name__ == "__main__":
    get_benchmark()()
