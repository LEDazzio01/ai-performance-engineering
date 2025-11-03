#!/usr/bin/env python3
"""Log stub mirroring the vLLM scheduler snippet cited in Chapter 16."""

import datetime
import logging
from enum import Enum, auto


class PreemptionMode(Enum):
    RECOMPUTE = auto()


def log_preemption() -> None:
    ts = datetime.datetime(2025, 5, 3, 14, 22, 7)
    logging.warning(
        "%s scheduler.py:1057 Sequence group 0 is preempted by %s because not enough KV cache space. "
        "total_cumulative_preemption_cnt =1",
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        PreemptionMode.RECOMPUTE,
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    log_preemption()


if __name__ == "__main__":
    main()

