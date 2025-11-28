"""Optimized NVSHMEM vs NCCL benchmark with NVLink5/NVLS knobs; skips on <2 GPUs."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
import torch
import torch.distributed as dist
from ch4.nccl_blackwell_config import (
    configure_nccl_for_8xB200,
    configure_nccl_for_blackwell,
    configure_nccl_for_gb200_gb300,
    detect_8xb200_topology,
)
from ch4.nvshmem_vs_nccl_benchmark import benchmark, init_distributed
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


def _configure_blackwell_nccl() -> None:
    try:
        topo = detect_8xb200_topology()
    except Exception:
        configure_nccl_for_blackwell(verbose=False)
        return

    if topo.get("has_grace_cpu"):
        configure_nccl_for_gb200_gb300(verbose=False)
    elif topo.get("num_gpus", 0) >= 8 and topo.get("is_8xb200"):
        configure_nccl_for_8xB200(verbose=False)
    else:
        configure_nccl_for_blackwell(verbose=False)


class OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU(BaseBenchmark):
    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_vs_nccl_benchmark requires >=2 GPUs")
        _configure_blackwell_nccl()

    def benchmark_fn(self) -> None:
        args = argparse.Namespace(
            min_bytes=4 * 1024,
            max_bytes=128 * 1024 * 1024,
            steps=7,
            iterations=75,
        )
        rank = init_distributed()
        try:
            results = benchmark(args)
            if rank == 0:
                print("\nNVSHMEM vs NCCL Benchmark (optimized for NVLink 5.0 / NVLS / TCE)")
                print("------------------------------------------------------------------")
                print(f"Symmetric memory available: {bool(results['nvshmem'])}")
                print("Message Size | NCCL Latency (us) | NCCL BW (GB/s) | NVSHMEM Latency (us) | NVSHMEM BW (GB/s)")
                print("-------------------------------------------------------------------------------------------")
                nvshmem_dict = {res.bytes: res for res in results["nvshmem"]}
                for res in results["nccl"]:
                    nv = nvshmem_dict.get(res.bytes)
                    nv_lat = f"{nv.latency_us:8.2f}" if nv else "   n/a "
                    nv_bw = f"{nv.bandwidth_gbps:8.2f}" if nv else "   n/a "
                    print(
                        f"{res.bytes:>12} | {res.latency_us:16.2f} | {res.bandwidth_gbps:13.2f} | "
                        f"{nv_lat:>18} | {nv_bw:>15}"
                    )
        finally:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=300)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"NVSHMEM vs NCCL optimized (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
