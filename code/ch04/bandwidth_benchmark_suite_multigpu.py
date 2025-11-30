"""Multi-GPU wrapper for bandwidth benchmark suite; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch04.bandwidth_benchmark_suite_8gpu import main as bandwidth_suite_main


def get_benchmark():
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")

    def _run():
        bandwidth_suite_main()

    return _run


if __name__ == "__main__":
    get_benchmark()()
