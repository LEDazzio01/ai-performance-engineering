"""Utilities for wrapping standalone CUDA binaries in the benchmark harness.

The chapter Makefiles build architecture-specific executables such as
``baseline_hbm3e_copy_sm100``. This helper compiles the requested binary (if
needed) and measures it by launching the executable from Python so the result
can participate in the standardized harness / metrics pipeline.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.harness.cuda_capabilities import pipeline_runtime_allowed
from core.benchmark.tma_checks import require_tma_instructions
from core.benchmark.timing_parser import parse_kernel_time_ms

ARCH_SUFFIX = {
    "sm_100": "_sm100",
    "sm_103": "_sm103",
    "sm_121": "_sm121",
}


def detect_supported_arch() -> str:
    """Infer the CUDA architecture for building binaries from the active GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to benchmark CUDA binaries")
    
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    if capability >= 121:
        return "sm_121"
    if capability >= 103:
        return "sm_103"
    if capability >= 100:
        return "sm_100"
    
    raise RuntimeError(
        f"Unsupported compute capability {major}.{minor}. "
        f"Supported architectures: {sorted(ARCH_SUFFIX)}"
    )


@dataclass
class BinaryRunResult:
    """Holds metrics parsed from a CUDA binary execution."""
    time_ms: Optional[float]
    raw_stdout: str
    raw_stderr: str


class CudaBinaryBenchmark(BaseBenchmark):
    """Benchmark wrapper that builds and runs a CUDA executable."""
    
    def __init__(
        self,
        chapter_dir: Path,
        binary_name: str,
        friendly_name: str,
        *,
        iterations: int = 3,
        warmup: int = 5,
        timeout_seconds: int = 15,
        run_args: Sequence[str] = (),
        requires_pipeline_api: bool = False,
        require_tma_instructions: bool = False,
        workload_params: Optional[dict] = None,
        time_regex: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.chapter_dir = chapter_dir
        self.binary_name = binary_name
        self.friendly_name = friendly_name
        self.iterations = iterations
        self.time_regex = time_regex  # Custom regex for parsing timing output
        self.warmup = warmup
        self.timeout_seconds = timeout_seconds
        self.run_args = list(run_args)
        self.requires_pipeline_api = requires_pipeline_api
        self.require_tma_instructions = require_tma_instructions
        self.use_reported_time = True
        self._workload_params = workload_params or {}
        
        self.arch: Optional[str] = None
        self.exec_path: Optional[Path] = None
        self._last_result: Optional[BinaryRunResult] = None
    
    # ------------------------------------------------------------------ Helper API
    def _build_binary(self) -> None:
        """Compile the requested CUDA binary if needed."""
        self.arch = detect_supported_arch()
        suffix = ARCH_SUFFIX[self.arch]
        target = f"{self.binary_name}{suffix}"
        build_cmd = ["make", f"ARCH={self.arch}", target]
        
        try:
            completed = subprocess.run(
                build_cmd,
                cwd=self.chapter_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout - CUDA compilation can take time for complex kernels
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Build timeout: {target} compilation exceeded 60 seconds")
        
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to build {target} (arch={self.arch}).\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        
        path = self.chapter_dir / target
        if not path.exists():
            raise FileNotFoundError(f"Built binary not found at {path}")
        self.exec_path = path
        if self.require_tma_instructions:
            require_tma_instructions(self.exec_path)
    
    def _run_once(self) -> BinaryRunResult:
        """Execute the compiled binary and parse its runtime."""
        if self.exec_path is None:
            raise RuntimeError("Executable path not set (build step missing)")
        
        try:
            completed = subprocess.run(
                [str(self.exec_path), *self.run_args],
                cwd=self.chapter_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Execution timeout: {self.exec_path.name} exceeded {self.timeout_seconds} seconds")
        
        if completed.returncode != 0:
            raise RuntimeError(
                f"{self.exec_path.name} exited with code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        
        time_ms = parse_kernel_time_ms(completed.stdout, self.time_regex)
        if time_ms is None:
            raise RuntimeError(
                f"Could not parse execution time from output of {self.exec_path.name}.\n"
                f"stdout:\n{completed.stdout}"
            )
        
        return BinaryRunResult(time_ms=time_ms, raw_stdout=completed.stdout, raw_stderr=completed.stderr)
    
    # ------------------------------------------------------------------ Benchmark API
    def setup(self) -> None:
        """Build the executable once before benchmarking."""
        if self.requires_pipeline_api:
            supported, reason = pipeline_runtime_allowed()
            if not supported:
                raise RuntimeError(f"SKIPPED: CUDA Pipeline API unavailable ({reason})")
        self._build_binary()
    
    def benchmark_fn(self) -> None:
        """Launch the executable and record its runtime."""
        self._last_result = self._run_once()
    
    def teardown(self) -> None:
        """No runtime resources to release."""
        pass
    
    def get_config(self) -> BenchmarkConfig:
        """Configure harness to use CPU timers (subprocess measures wall time)."""
        return BenchmarkConfig(
            iterations=self.iterations,
            warmup=self.warmup,
            timeout_seconds=self.timeout_seconds,
            device=torch.device("cuda"),
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Ensure we captured at least one timing measurement."""
        if self._last_result is None:
            return "Binary did not execute"
        if self._last_result.time_ms is None or self._last_result.time_ms <= 0:
            return f"Invalid runtime parsed: {self._last_result.time_ms} ms"
        return None
    
    def get_input_signature(self) -> Optional[dict]:
        """Return input signature for verification.
        
        CUDA binaries have workload parameters baked in at compile time.
        Subclasses should pass workload_params to __init__ or override this method.
        
        Common parameters to include:
        - N, M, K: Matrix/vector sizes
        - batch_size, seq_len: Sequence dimensions
        - hidden_dim, num_heads: Model dimensions
        """
        if self._workload_params:
            return self._workload_params
        # Return empty dict to indicate no explicit workload params
        # binary_name is excluded from comparison by the harness since it's
        # expected to differ between baseline and optimized
        return {}
    
    # Convenience accessors -----------------------------------------------------
    @property
    def last_time_ms(self) -> Optional[float]:
        return None if self._last_result is None else self._last_result.time_ms
    
    @property
    def last_stdout(self) -> Optional[str]:
        return None if self._last_result is None else self._last_result.raw_stdout
