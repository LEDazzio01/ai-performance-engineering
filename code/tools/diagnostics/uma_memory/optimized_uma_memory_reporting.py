"""UMA-aware memory reporting that accounts for reclaimable DRAM + SWAP."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import socket
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from ch2.uma_memory_utils import format_bytes, is_integrated_gpu, read_meminfo


class OptimizedUmaMemoryReportingBenchmark(BaseBenchmark):
    """Adjusts cudaMemGetInfo with host MemAvailable + reclaimable swap on UMA."""

    def __init__(self, reclaim_fraction: float = 0.9):
        super().__init__()
        self.reclaim_fraction = reclaim_fraction
        self.cuda_free_bytes = 0
        self.cuda_total_bytes = 0
        self.mem_available_bytes = 0
        self.swap_free_bytes = 0
        self.allocatable_bytes = 0
        self.per_process_bytes: Optional[int] = None

    def setup(self) -> None:
        torch.cuda.empty_cache()
        self._sample()

    def _sample(self) -> None:
        free, total = torch.cuda.mem_get_info()
        self.cuda_free_bytes = free
        self.cuda_total_bytes = total
        snapshot = read_meminfo()
        if snapshot:
            self.mem_available_bytes = snapshot.effective_available_kb() * 1024
            self.swap_free_bytes = snapshot.swap_free_kb * 1024
            self.allocatable_bytes = snapshot.allocatable_bytes(self.reclaim_fraction)
        else:
            self.mem_available_bytes = 0
            self.swap_free_bytes = 0
            self.allocatable_bytes = free
        self.per_process_bytes = self._collect_per_process_usage()

    def _collect_per_process_usage(self) -> Optional[int]:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                memory_used = sum(p.usedGpuMemory for p in procs if p.usedGpuMemory is not None)
            finally:
                pynvml.nvmlShutdown()
            return int(memory_used)
        except Exception:
            return None

    def benchmark_fn(self) -> None:
        self._sample()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            enable_memory_tracking=False,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        metrics: Dict[str, float] = {
            "cuda_free_gb": self.cuda_free_bytes / (1024**3),
            "uma_allocatable_gb": self.allocatable_bytes / (1024**3),
            "memavailable_gb": self.mem_available_bytes / (1024**3),
            "swapfree_gb": self.swap_free_bytes / (1024**3),
            "reclaim_fraction": float(self.reclaim_fraction),
        }
        if self.per_process_bytes is not None:
            metrics["per_process_usage_gb"] = self.per_process_bytes / (1024**3)
        return metrics


    def snapshot_dict(self) -> Dict[str, object]:
        """Produce a JSON-serializable snapshot of the current UMA estimate."""
        device_name = None
        compute_capability = None
        total_mem_gb = None
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        compute_capability = f"{props.major}.{props.minor}"
        total_mem_gb = props.total_memory / (1024**3)

        return {
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda if torch.version else None,
            "device_name": device_name,
            "compute_capability": compute_capability,
            "total_mem_gb": total_mem_gb,
            "integrated_gpu": is_integrated_gpu(),
            "reclaim_fraction": float(self.reclaim_fraction),
            "cuda_free_bytes": int(self.cuda_free_bytes),
            "cuda_total_bytes": int(self.cuda_total_bytes),
            "memavailable_bytes": int(self.mem_available_bytes),
            "swapfree_bytes": int(self.swap_free_bytes),
            "uma_allocatable_bytes": int(self.allocatable_bytes),
            "per_process_bytes": int(self.per_process_bytes) if self.per_process_bytes is not None else None,
        }

    def write_snapshot(self, snapshot_dir: Path, file_name: Optional[str] = None) -> Path:
        """Write snapshot JSON to the given directory."""
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            file_name = f"uma_snapshot_{timestamp}.json"
        path = snapshot_dir / file_name
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.snapshot_dict(), f, indent=2)
        return path


def summarize(reclaim_fraction: float = 0.9) -> OptimizedUmaMemoryReportingBenchmark:
    bench = OptimizedUmaMemoryReportingBenchmark(reclaim_fraction=reclaim_fraction)
    bench.setup()
    bench.benchmark_fn()
    integrated = is_integrated_gpu()
    print("\n=== UMA-aware CUDA memory report ===")
    print(f"Integrated GPU detected: {integrated}")
    print(f"cudaMemGetInfo free:        {format_bytes(bench.cuda_free_bytes)}")
    print(f"cudaMemGetInfo total:       {format_bytes(bench.cuda_total_bytes)}")
    print(f"Host MemAvailable:          {format_bytes(bench.mem_available_bytes)}")
    print(f"SwapFree (reclaimable {bench.reclaim_fraction:.0%}): {format_bytes(bench.swap_free_bytes)}")
    print(f"UMA allocatable estimate:   {format_bytes(bench.allocatable_bytes)}")
    if bench.per_process_bytes is not None:
        print(f"Per-process NVML usage sum: {format_bytes(bench.per_process_bytes)}")
    return bench


def get_benchmark() -> BaseBenchmark:
    return OptimizedUmaMemoryReportingBenchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMA-aware memory reporting with optional snapshot.")
    parser.add_argument(
        "--reclaim-fraction",
        type=float,
        default=0.9,
        help="Fraction of SwapFree to treat as reclaimable for UMA estimation.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Write a JSON snapshot to artifacts/uma_memory_snapshots.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("artifacts/uma_memory_snapshots"),
        help="Directory for UMA snapshot output.",
    )
    args = parser.parse_args()

    bench = summarize(reclaim_fraction=args.reclaim_fraction)
    if args.snapshot:
        out_path = bench.write_snapshot(args.snapshot_dir)
        print(f"\nSnapshot written: {out_path}")
