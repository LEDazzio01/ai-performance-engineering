"""CLI commands for microbenchmarks (disk, PCIe, memory, tensor core, SFU, loopback)."""

from __future__ import annotations

import json
from typing import Any


def _print_json(data: Any):
    print(json.dumps(data, indent=2, default=str))


def disk(args) -> int:
    from tools import microbench
    res = microbench.disk_io_test(
        file_size_mb=args.file_size_mb,
        block_size_kb=args.block_size_kb,
        tmp_dir=args.tmp_dir,
    )
    _print_json(res)
    return 0


def pcie(args) -> int:
    from tools import microbench
    res = microbench.pcie_bandwidth_test(size_mb=args.size_mb, iters=args.iters)
    _print_json(res)
    return 0


def mem_hierarchy(args) -> int:
    from tools import microbench
    res = microbench.mem_hierarchy_test(size_mb=args.size_mb, stride=args.stride)
    _print_json(res)
    return 0


def tensor_core(args) -> int:
    from tools import microbench
    res = microbench.tensor_core_bench(size=args.size, precision=args.precision)
    _print_json(res)
    return 0


def sfu(args) -> int:
    from tools import microbench
    res = microbench.sfu_bench(size=args.elements)
    _print_json(res)
    return 0


def loopback(args) -> int:
    from tools import microbench
    res = microbench.network_loopback_test(size_mb=args.size_mb, port=args.port)
    _print_json(res)
    return 0

