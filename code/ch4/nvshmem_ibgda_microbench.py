"""Shared helpers for the NVSHMEM IBGDA microbenchmark."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from common.python.benchmark_harness import BenchmarkConfig
from common.python.cuda_binary_benchmark import BinaryRunResult, CudaBinaryBenchmark


def _default_symmetric_size() -> str:
    """Best-effort symmetric heap sizing for the microbench."""
    return os.getenv("NVSHMEM_SYMMETRIC_SIZE", "128M")


class NvshmemIbgdaMicrobench(CudaBinaryBenchmark):
    """Wrap the nvshmem_ibgda_microbench CUDA binary for the harness."""

    def __init__(
        self,
        *,
        enable_ibgda: bool,
        mode: str = "p",
        bytes_per_message: int = 1024,
        ctas: int = 32,
        threads: int = 256,
        iters: int = 500,
        world_size: int = 2,
        symmetric_size: str = _default_symmetric_size(),
    ) -> None:
        self.enable_ibgda = enable_ibgda
        self.mode = mode
        self.bytes_per_message = bytes_per_message
        self.ctas = ctas
        self.threads = threads
        self.iters = iters
        self.world_size = max(1, world_size)
        self.symmetric_size = symmetric_size
        self.nvshmemrun: Optional[str] = None
        self._parsed_metrics: Dict[str, float] = {}

        args = [
            f"--mode={mode}",
            f"--bytes={bytes_per_message}",
            f"--ctas={ctas}",
            f"--threads={threads}",
            f"--iters={iters}",
        ]

        super().__init__(
            chapter_dir=Path(__file__).parent,
            binary_name="nvshmem_ibgda_microbench",
            friendly_name=f"nvshmem_ibgda_{mode}",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            run_args=args,
            time_regex=None,  # Use harness timing instead of stdout parsing.
        )

    # --------------------------------------------------------------------- Setup/teardown
    def setup(self) -> None:
        if torch.cuda.device_count() < self.world_size:
            raise RuntimeError("SKIPPED: nvshmem_ibgda_microbench requires >=2 GPUs")

        lib_candidates = [
            Path(os.getenv("NVSHMEM_HOME", "")) / "libnvshmem_host.so",
            Path("/usr/lib/x86_64-linux-gnu/libnvshmem_host.so"),
            Path("/usr/lib/aarch64-linux-gnu/libnvshmem_host.so"),
        ]
        if not any(p.exists() for p in lib_candidates):
            raise RuntimeError(
                "SKIPPED: NVSHMEM not installed (libnvshmem_host.so not found; set NVSHMEM_HOME)"
            )

        # Prefer nvshmemrun from PATH, fall back to NVSHMEM_HOME/bin.
        launcher = shutil.which("nvshmemrun")
        if launcher is None:
            nvshmem_home = os.getenv("NVSHMEM_HOME")
            candidate = (
                Path(nvshmem_home) / "bin" / "nvshmemrun"
                if nvshmem_home
                else None
            )
            if candidate and candidate.exists():
                launcher = str(candidate)

        if launcher is None:
            raise RuntimeError("SKIPPED: nvshmemrun not found in PATH or NVSHMEM_HOME/bin")

        self.nvshmemrun = launcher

        try:
            super().setup()
        except Exception as exc:
            raise RuntimeError(f"SKIPPED: failed to build nvshmem_ibgda_microbench ({exc})") from exc

    # --------------------------------------------------------------------- Runtime helpers
    def _runtime_env(self) -> Dict[str, str]:
        env = {
            "NVSHMEM_SYMMETRIC_SIZE": self.symmetric_size,
        }
        if self.enable_ibgda:
            env.update(
                {
                    "NVSHMEM_IB_ENABLE_IBGDA": "1",
                    "NVSHMEM_IBGDA_NIC_HANDLER": "gpu",
                    "NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE": "gpumem",
                    "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH": "1",
                }
            )
        else:
            env.update(
                {
                    "NVSHMEM_IB_ENABLE_IBGDA": "0",
                    "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH": "32",
                }
            )
        return env

    def _parse_metrics(self, stdout: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        bw = re.search(r"bw=([0-9.]+)\s*GB/s", stdout)
        if bw:
            metrics["bandwidth_gbps"] = float(bw.group(1))
        mops = re.search(r"rate=([0-9.]+)\s*MOPS", stdout)
        if mops:
            metrics["mops"] = float(mops.group(1))
        return metrics

    def _run_once(self) -> BinaryRunResult:
        if self.exec_path is None:
            raise RuntimeError("Executable path not set (build step missing)")
        if self.nvshmemrun is None:
            raise RuntimeError("nvshmemrun launcher not resolved")

        env = os.environ.copy()
        env.update(self._runtime_env())

        cmd = [
            self.nvshmemrun,
            "-np",
            str(self.world_size),
            str(self.exec_path),
            *self.run_args,
        ]

        start = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=self.chapter_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=env,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.binary_name} exited with code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        self._parsed_metrics = self._parse_metrics(completed.stdout)
        if self.world_size <= 1 and not self._parsed_metrics:
            # Single-PE smoke test: no metrics emitted.
            self._parsed_metrics = {"single_pe": 1.0}
        self._parsed_metrics["elapsed_ms"] = elapsed_ms
        return BinaryRunResult(time_ms=elapsed_ms, raw_stdout=completed.stdout, raw_stderr=completed.stderr)

    # --------------------------------------------------------------------- Benchmark API
    def benchmark_fn(self) -> None:
        self._last_result = self._run_once()

    def get_config(self) -> BenchmarkConfig:
        # Single execution; wall-clock timer is provided by harness.
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=self.timeout_seconds,
            setup_timeout_seconds=60,
            enable_memory_tracking=False,
            multi_gpu_required=True,
        )

    def validate_result(self) -> Optional[str]:
        if self._last_result is None:
            return "Binary did not execute"
        if not self._parsed_metrics:
            if self.world_size <= 1:
                return None
            return "Failed to parse metrics from nvshmem_ibgda_microbench output"
        return None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._parsed_metrics
