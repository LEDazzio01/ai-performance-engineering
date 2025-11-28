"""Microbenchmark for H2D transfers across pinned/pageable/NUMA-host slabs."""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from typing import Dict, Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    """Resolve CUDA device for measurement."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for KV locality microbench")
    return torch.device("cuda")


class _NumaHelper:
    """Best-effort libnuma wrapper used to move pinned slabs between nodes."""

    def __init__(self) -> None:
        self._lib = self._load()
        self.nodes = self._detect_nodes()
        self.preferred = self.nodes[0] if self.nodes else None

    def _load(self) -> Optional[ctypes.CDLL]:
        for lib in ("libnuma.so.1", "libnuma.so"):
            try:
                handle = ctypes.CDLL(lib)
            except OSError:
                continue
            try:
                if handle.numa_available() < 0:  # type: ignore[attr-defined]
                    continue
            except Exception:
                continue
            return handle
        return None

    def _detect_nodes(self) -> list[int]:
        root = "/sys/devices/system/node"
        if not os.path.isdir(root):
            return []
        nodes: list[int] = []
        for entry in os.listdir(root):
            if entry.startswith("node") and entry[4:].isdigit():
                nodes.append(int(entry[4:]))
        return sorted(nodes)

    def move_to_node(self, tensor: torch.Tensor, node: int) -> bool:
        if self._lib is None:
            return False
        try:
            ptr = ctypes.c_void_p(tensor.data_ptr())
            length = ctypes.c_size_t(tensor.numel() * tensor.element_size())
            res = self._lib.numa_tonode_memory(ptr, length, ctypes.c_int(node))  # type: ignore[attr-defined]
            return res == 0
        except Exception:
            return False

    def remote_node(self) -> Optional[int]:
        if len(self.nodes) < 2:
            return None
        if self.preferred is None:
            return self.nodes[0]
        for n in self.nodes:
            if n != self.preferred:
                return n
        return None


class KvLocalityMicrobench(BaseBenchmark):
    """Compare H2D copy time for HBM vs pinned-local vs pinned-remote vs pageable."""

    def __init__(self) -> None:
        super().__init__()
        args = _CLI_ARGS
        self.rows = args.rows
        self.cols = args.cols
        self.iters = args.iters
        self.device = None
        self.dst = None
        self.pageable = None
        self.pinned_local = None
        self.pinned_remote = None
        self.hbm = None
        self.helper = _NumaHelper()
        self.results: Dict[str, float] = {}

    def setup(self) -> None:
        self.device = resolve_device()
        shape = (self.rows, self.cols)
        self.dst = torch.empty(shape, device=self.device, dtype=torch.float16)
        self.hbm = torch.empty(shape, device=self.device, dtype=torch.float16)
        self.pageable = torch.randn(shape, device="cpu", dtype=torch.float16)
        self.pinned_local = torch.randn(shape, device="cpu", dtype=torch.float16, pin_memory=True)
        if self.helper.preferred is not None:
            self.helper.move_to_node(self.pinned_local, self.helper.preferred)

        remote_node = self.helper.remote_node()
        if remote_node is not None:
            self.pinned_remote = torch.randn(shape, device="cpu", dtype=torch.float16, pin_memory=True)
            self.helper.move_to_node(self.pinned_remote, remote_node)

    def _bench_copy(self, src: torch.Tensor) -> float:
        assert self.dst is not None
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.iters):
            with torch.cuda.stream(stream):
                self.dst.copy_(src, non_blocking=True)
        stream.synchronize()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / float(self.iters)

    def benchmark_fn(self) -> None:
        if any(x is None for x in [self.dst, self.pageable, self.pinned_local, self.hbm]):
            raise RuntimeError("Buffers not initialized")

        self.results["hbm_to_hbm_ms"] = self._bench_copy(self.hbm)  # type: ignore[arg-type]
        self.results["pageable_to_hbm_ms"] = self._bench_copy(self.pageable)  # type: ignore[arg-type]
        self.results["pinned_local_to_hbm_ms"] = self._bench_copy(self.pinned_local)  # type: ignore[arg-type]
        if self.pinned_remote is not None:
            self.results["pinned_remote_to_hbm_ms"] = self._bench_copy(self.pinned_remote)

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self.results:
            return None
        metrics = {k: float(v * 1000.0) for k, v in self.results.items()}  # seconds -> ms
        metrics["shape_rows"] = float(self.rows)
        metrics["shape_cols"] = float(self.cols)
        if self.helper.preferred is not None:
            metrics["numa_local_node"] = float(self.helper.preferred)
        remote = self.helper.remote_node()
        if remote is not None:
            metrics["numa_remote_node"] = float(remote)
        return metrics

    def teardown(self) -> None:
        self.dst = None
        self.pageable = None
        self.pinned_local = None
        self.pinned_remote = None
        self.hbm = None
        self.results = {}
        torch.cuda.empty_cache()


def get_benchmark() -> BaseBenchmark:
    return KvLocalityMicrobench()


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rows", type=int, default=256, help="Rows of the test tensor.")
    parser.add_argument("--cols", type=int, default=4096, help="Cols of the test tensor.")
    parser.add_argument("--iters", type=int, default=200, help="Iterations per copy variant.")
    return parser.parse_known_args()[0]


_CLI_ARGS = _parse_cli_args()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.setup()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
