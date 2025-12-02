"""Centralized default values for benchmark configuration.

This module provides a single source of truth for all default values used
throughout the benchmark harness, enabling easy configuration via environment
variables and configuration files.

CRITICAL: Warmup iterations are REQUIRED to ensure JIT/compile overhead is NOT
included in measurements. torch.compile, Triton kernels, and CUDA JIT all require
warmup to reach steady-state performance.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# WARMUP REQUIREMENTS - DO NOT CHANGE WITHOUT UNDERSTANDING THE IMPLICATIONS
# =============================================================================
# These constants ensure accurate benchmark measurements by excluding JIT overhead.
# torch.compile typically needs 1-3 calls to fully compile, but we use higher values
# to account for:
#   - Multiple recompilations for dynamic shapes
#   - Triton kernel JIT compilation
#   - CUDA driver initialization
#   - cuDNN autotuning
#   - Memory allocator warmup

MINIMUM_WARMUP_ITERATIONS = 5  # Absolute minimum for ANY benchmark
RECOMMENDED_WARMUP_TORCH_COMPILE = 10  # For benchmarks using torch.compile
RECOMMENDED_WARMUP_TRITON = 10  # For benchmarks using Triton kernels
RECOMMENDED_WARMUP_CUDA_GRAPHS = 10  # For benchmarks using CUDA graphs
DEFAULT_WARMUP_ITERATIONS = 10  # Standard default for all benchmarks


def validate_warmup(warmup: int, context: str = "") -> int:
    """Validate warmup iterations and warn if too low.
    
    Args:
        warmup: Requested warmup iterations
        context: Optional context for warning messages (e.g., benchmark name)
        
    Returns:
        Validated warmup value (raised to minimum if necessary)
    """
    if warmup < MINIMUM_WARMUP_ITERATIONS:
        ctx_str = f" for {context}" if context else ""
        warnings.warn(
            f"Warmup iterations ({warmup}){ctx_str} is below minimum ({MINIMUM_WARMUP_ITERATIONS}). "
            f"This may include JIT/compile overhead in measurements, causing inaccurate results. "
            f"Automatically raising to {MINIMUM_WARMUP_ITERATIONS}. "
            f"If you intentionally need warmup=0, add internal warmup in setup() instead.",
            UserWarning,
            stacklevel=3
        )
        return MINIMUM_WARMUP_ITERATIONS
    return warmup


@dataclass
class BenchmarkDefaults:
    """Centralized default values for benchmark configuration.
    
    All defaults can be overridden by passing values directly to BenchmarkConfig
    or via CLI flags (e.g., --iterations, --warmup, --profile).
    
    Note: Environment variables are no longer supported. Use CLI flags instead.
    
    IMPORTANT: Warmup iterations should NEVER be set below MINIMUM_WARMUP_ITERATIONS
    to ensure JIT/compile overhead is excluded from measurements.
    """
    
    # Execution defaults
    iterations: int = 100
    warmup: int = DEFAULT_WARMUP_ITERATIONS  # Use constant for consistency
    min_run_time_ms: float = 100.0
    
    # Statistics defaults
    percentiles: List[float] = field(default_factory=lambda: [25, 50, 75, 99])
    
    # Feature flags
    enable_memory_tracking: bool = False
    enable_profiling: bool = False
    enable_nsys: bool = False
    enable_ncu: bool = True
    enable_proton: bool = False
    enable_nvtx: Optional[bool] = None  # Auto-enabled if profiling enabled
    # Free allocator state between benchmarks by default.
    enable_cleanup: bool = True
    # Prefer subprocess isolation by default; fall back to thread mode when explicitly requested.
    use_subprocess: bool = True
    execution_mode: Optional[str] = None
    launch_via: str = "python"  # python | torchrun
    nproc_per_node: Optional[int] = None
    nnodes: Optional[str] = None
    rdzv_backend: Optional[str] = None
    rdzv_endpoint: Optional[str] = None
    env_passthrough: List[str] = field(default_factory=lambda: ["CUDA_VISIBLE_DEVICES"])
    target_extra_args: dict = field(default_factory=dict)
    multi_gpu_required: bool = False
    profile_type: str = "minimal"
    nsys_nvtx_include: Optional[List[str]] = None
    
    # Reproducibility defaults
    # Default to nondeterministic for performance; users can opt-in if needed.
    deterministic: bool = False
    seed: Optional[int] = 1337
    
    # Timeout defaults (in seconds)
    # Increased setup timeout for CUDA JIT compilation (can take 60+ seconds)
    setup_timeout_seconds: Optional[int] = 300
    warmup_timeout_seconds: Optional[int] = None  # Defaults to measurement_timeout
    # Allow slow first-time compilations in subprocess isolation.
    measurement_timeout_seconds: int = 1200
    profiling_timeout_seconds: Optional[int] = None  # Defaults to max(nsys, ncu)
    nsys_timeout_seconds: int = 180
    ncu_timeout_seconds: int = 300
    proton_timeout_seconds: int = 180
    timeout_multiplier: float = 3.0
    ncu_sampling_interval: int = 75000
    
    # Legacy timeout (deprecated)
    timeout_seconds: int = 900
    
    # Output defaults
    profiling_output_dir: Optional[str] = None
    ncu_metric_set: str = "auto"  # 'auto', 'deep_dive', 'roofline', 'minimal'
    
    @classmethod
    def from_env(cls) -> BenchmarkDefaults:
        """Create BenchmarkDefaults with default values.
        
        Note: Environment variables are no longer supported. Use CLI flags instead.
        This method now simply returns default values for backward compatibility.
        """
        return cls()
    
    def to_dict(self) -> dict:
        """Convert defaults to dictionary."""
        return {
            "iterations": self.iterations,
            "warmup": self.warmup,
            "min_run_time_ms": self.min_run_time_ms,
            "percentiles": self.percentiles,
            "enable_memory_tracking": self.enable_memory_tracking,
            "enable_profiling": self.enable_profiling,
            "enable_nsys": self.enable_nsys,
            "enable_ncu": self.enable_ncu,
            "enable_proton": self.enable_proton,
            "enable_nvtx": self.enable_nvtx,
            "enable_cleanup": self.enable_cleanup,
            "use_subprocess": self.use_subprocess,
            "execution_mode": self.execution_mode,
            "launch_via": self.launch_via,
            "nproc_per_node": self.nproc_per_node,
            "nnodes": self.nnodes,
            "rdzv_backend": self.rdzv_backend,
            "rdzv_endpoint": self.rdzv_endpoint,
            "env_passthrough": self.env_passthrough,
            "target_extra_args": self.target_extra_args,
            "multi_gpu_required": self.multi_gpu_required,
            "profile_type": self.profile_type,
            "nsys_nvtx_include": self.nsys_nvtx_include,
            "deterministic": self.deterministic,
            "seed": self.seed,
            "setup_timeout_seconds": self.setup_timeout_seconds,
            "warmup_timeout_seconds": self.warmup_timeout_seconds,
            "measurement_timeout_seconds": self.measurement_timeout_seconds,
            "profiling_timeout_seconds": self.profiling_timeout_seconds,
            "nsys_timeout_seconds": self.nsys_timeout_seconds,
            "ncu_timeout_seconds": self.ncu_timeout_seconds,
            "proton_timeout_seconds": self.proton_timeout_seconds,
            "timeout_multiplier": self.timeout_multiplier,
            "timeout_seconds": self.timeout_seconds,
            "profiling_output_dir": self.profiling_output_dir,
            "ncu_metric_set": self.ncu_metric_set,
            "ncu_sampling_interval": self.ncu_sampling_interval,
        }


# Global instance - can be overridden for testing or custom configurations
_defaults = BenchmarkDefaults.from_env()


def get_defaults() -> BenchmarkDefaults:
    """Get the global BenchmarkDefaults instance."""
    return _defaults


def set_defaults(defaults: BenchmarkDefaults) -> None:
    """Set the global BenchmarkDefaults instance (useful for testing)."""
    global _defaults
    _defaults = defaults


def get_minimum_warmup() -> int:
    """Get the minimum allowed warmup iterations."""
    return MINIMUM_WARMUP_ITERATIONS


def get_recommended_warmup(uses_torch_compile: bool = False, 
                            uses_triton: bool = False,
                            uses_cuda_graphs: bool = False) -> int:
    """Get recommended warmup iterations based on features used.
    
    Args:
        uses_torch_compile: Whether benchmark uses torch.compile
        uses_triton: Whether benchmark uses Triton kernels
        uses_cuda_graphs: Whether benchmark uses CUDA graphs
        
    Returns:
        Recommended warmup iterations
    """
    if uses_torch_compile:
        return RECOMMENDED_WARMUP_TORCH_COMPILE
    if uses_triton:
        return RECOMMENDED_WARMUP_TRITON
    if uses_cuda_graphs:
        return RECOMMENDED_WARMUP_CUDA_GRAPHS
    return DEFAULT_WARMUP_ITERATIONS
