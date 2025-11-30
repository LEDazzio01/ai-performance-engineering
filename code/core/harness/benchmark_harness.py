"""Production-grade benchmarking harness with profiling integration.

Provides industry-standard benchmarking using Triton do_bench, PyTorch Timer,
and custom CUDA Events. Supports nsys, ncu, and PyTorch profiler integration.
"""

from __future__ import annotations

import copy
import gc
import json
import importlib
import inspect
import os
import random
import re
import shutil
import statistics
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union, cast

import numpy as np
import torch

from core.utils.compile_utils import enable_tf32

if TYPE_CHECKING:
    from core.benchmark.models import (
        BenchmarkResult as PydanticBenchmarkResult,
        BenchmarkRun,
        MemoryStats,
        TimingStats,
        InferenceTimingStats,
        ProfilerArtifacts,
        ProfilerMetrics,
        NsysMetrics,
        NcuMetrics,
        ProtonMetrics,
        TorchMetrics,
    )

# Pydantic is required - fail fast if not available
from core.benchmark.models import (
    BenchmarkResult as PydanticBenchmarkResult,
    BenchmarkRun,
    MemoryStats,
    TimingStats,
    InferenceTimingStats,
    ProfilerArtifacts,
    ProfilerMetrics,
    NsysMetrics,
    NcuMetrics,
    ProtonMetrics,
    TorchMetrics,
    ThroughputStats,
)

PYDANTIC_AVAILABLE = True

# Import unified profiling runner
try:
    from core.profiling.profiling_runner import (
        run_nsys_profiling,
        run_ncu_profiling,
        check_nsys_available,
        check_ncu_available,
        run_proton_profiling,
        check_proton_available,
    )
    PROFILING_RUNNER_AVAILABLE = True
except ImportError:
    PROFILING_RUNNER_AVAILABLE = False
    def run_nsys_profiling(
        benchmark: Any,
        benchmark_module: Any,
        benchmark_class: str,
        output_dir: Path,
        config: Any,
        profiler_config: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
        metrics: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        raise ImportError("profiling_runner not available")
    
    def run_ncu_profiling(
        benchmark: Any,
        benchmark_module: Any,
        benchmark_class: str,
        output_dir: Path,
        config: Any,
        profiler_config: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
        metrics: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        raise ImportError("profiling_runner not available")
    
    def check_nsys_available() -> bool:
        return False
    
    def check_ncu_available() -> bool:
        return False
    
    def run_proton_profiling(
        benchmark: Any,
        benchmark_module: Any,
        benchmark_class: str,
        output_dir: Path,
        config: Any,
        profiler_config: Optional[Any] = None,
        timeout_seconds: Optional[int] = None,
        metrics: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        raise ImportError("profiling_runner not available")
    
    def check_proton_available() -> bool:
        return False

# Import profiler wrapper
try:
    from core.profiling.profiler_wrapper import create_benchmark_wrapper
    PROFILER_WRAPPER_AVAILABLE = True
except ImportError:
    PROFILER_WRAPPER_AVAILABLE = False

# Import logger
try:
    from core.utils.logger import get_logger
    logger = get_logger(__name__)
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

_QUICK_WINS_CONFIGURED = False
_SDPA_KERNEL_CONTEXT = None


def _configure_quick_wins() -> None:
    """Apply global TF32 configuration once per process."""
    global _QUICK_WINS_CONFIGURED
    if _QUICK_WINS_CONFIGURED:
        return
    if torch.cuda.is_available():
        try:
            enable_tf32()
        except Exception as exc:  # pragma: no cover - defensive
            if LOGGER_AVAILABLE:
                logger.warning("TF32 enablement failed: %s", exc)
        _configure_attention_kernels()
        _configure_matmul_reduction()
    _QUICK_WINS_CONFIGURED = True


def _configure_attention_kernels() -> None:
    """Prefer flash/efficient SDPA backends when available."""
    global _SDPA_KERNEL_CONTEXT
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover - API availability differs by PyTorch
        sdpa_kernel = None  # type: ignore[assignment]
        SDPBackend = None  # type: ignore[assignment]

    if sdpa_kernel is not None and SDPBackend is not None:
        preferred = [
            getattr(SDPBackend, name)
            for name in ("TRANSFORMER_ENGINE", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN")
            if hasattr(SDPBackend, name)
        ]
        if preferred:
            try:
                ctx = sdpa_kernel(preferred)
                ctx.__enter__()  # keep context active for process lifetime
                _SDPA_KERNEL_CONTEXT = ctx
                return
            except Exception as exc:  # pragma: no cover - fallback below
                if LOGGER_AVAILABLE:
                    logger.debug("SDPA kernel preference failed: %s", exc)

    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
        torch.backends.cuda.enable_math_sdp(False)  # type: ignore[attr-defined]


def _configure_matmul_reduction() -> None:
    """Enable reduced-precision reductions to hit fused matmul fast paths."""
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # type: ignore[attr-defined]
    except Exception:
        return


def _is_chapter_or_labs_benchmark(benchmark: Any) -> bool:
    """Return True if the benchmark originates from any chapter directory (chXX/) or labs/."""
    try:
        file_path = inspect.getfile(benchmark.__class__)
    except Exception:
        return False
    normalized = str(Path(file_path)).replace("\\", "/")
    return "/labs/" in normalized or "/ch" in normalized


_configure_quick_wins()


def _extract_skip_reason_from_messages(messages: Sequence[str]) -> Optional[str]:
    """Return the SKIPPED reason embedded in harness error messages."""
    for msg in messages:
        upper = msg.upper()
        if "SKIPPED" not in upper:
            continue
        if "SKIPPED:" in msg:
            return msg.split("SKIPPED:", 1)[1].strip()
        idx = upper.find("SKIPPED")
        if idx != -1:
            return msg[idx:].strip()
    return None

from core.profiling.gpu_memory_logger import (
    GpuMemoryLogger,
    resolve_gpu_log_interval,
    resolve_gpu_log_path,
    should_enable_gpu_memory_logging,
)
from core.profiling.gpu_telemetry import query_gpu_telemetry


class BenchmarkMode(Enum):
    """Benchmarking mode selection."""
    TRITON = "triton"  # Use triton.testing.do_bench
    PYTORCH = "pytorch"  # Use torch.utils.benchmark.Timer
    CUSTOM = "custom"  # Use CUDA Events / time.perf_counter
    TRAINING = "training"  # Alias for CUSTOM; kept for backward compatibility
    INFERENCE = "inference"  # Alias for CUSTOM; kept for backward compatibility


class ExecutionMode(str, Enum):
    """How the harness executes benchmarks."""
    SUBPROCESS = "subprocess"
    THREAD = "thread"


class LaunchVia(str, Enum):
    """Which launcher to use for running a benchmark."""
    PYTHON = "python"
    TORCHRUN = "torchrun"


# Import BenchmarkDefaults to use as source of truth for defaults
try:
    from core.benchmark.defaults import BenchmarkDefaults, get_defaults
    _get_defaults_fn: Optional[Callable[[], BenchmarkDefaults]] = get_defaults
except ImportError:
    # Fallback if benchmark defaults are unavailable (docs builds)
    _get_defaults_fn = None


def _get_default_value(attr_name: str, fallback):
    """Get default value from BenchmarkDefaults, with fallback."""
    if _get_defaults_fn is None:
        return fallback
    defaults = _get_defaults_fn()
    return getattr(defaults, attr_name, fallback)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.
    
    Default values are loaded from BenchmarkDefaults, which provides hardcoded defaults.
    This ensures a single source of truth for all configuration defaults.
    Configuration can be overridden via CLI flags or by passing values directly.
    
    Reproducibility Notes:
        - deterministic=True: Enables deterministic algorithms (torch.use_deterministic_algorithms).
          This ensures bitwise reproducibility but may impact performance:
          * cuDNN algorithms may be slower (no autotuning)
          * Some operations may use less efficient deterministic implementations
          * Operations without deterministic implementations may raise errors at runtime
          * Performance impact typically 5-20% depending on workload
          * Use only when reproducibility is more important than performance
        
        - seed: Sets random seeds for Python random, numpy, torch, and CUDA.
          When combined with deterministic=True, ensures reproducible results.
          Without deterministic=True, results may still vary due to non-deterministic algorithms.
    """
    # Use field(default_factory=...) to get defaults from BenchmarkDefaults at instance creation time
    # Each lambda calls _get_default_value() which calls get_defaults() to get current defaults
    iterations: int = field(default_factory=lambda: _get_default_value("iterations", 100))
    warmup: int = field(default_factory=lambda: _get_default_value("warmup", 10))
    min_run_time_ms: float = field(default_factory=lambda: _get_default_value("min_run_time_ms", 100.0))
    percentiles: List[float] = field(default_factory=lambda: (_get_default_value("percentiles", [25, 50, 75, 90, 99]) or [25, 50, 75, 90, 99]).copy())
    enable_memory_tracking: bool = field(default_factory=lambda: _get_default_value("enable_memory_tracking", False))
    deterministic: bool = field(default_factory=lambda: _get_default_value("deterministic", False))
    seed: Optional[int] = field(default_factory=lambda: _get_default_value("seed", None))
    device: Optional[torch.device] = None  # Not in BenchmarkDefaults (runtime-specific)
    enable_profiling: bool = field(default_factory=lambda: _get_default_value("enable_profiling", False))
    enable_nsys: bool = field(default_factory=lambda: _get_default_value("enable_nsys", False))
    enable_ncu: bool = field(default_factory=lambda: _get_default_value("enable_ncu", False))
    enable_proton: bool = field(default_factory=lambda: _get_default_value("enable_proton", False))
    profiling_output_dir: Optional[str] = field(default_factory=lambda: _get_default_value("profiling_output_dir", None))
    profile_mode: Optional[str] = field(default_factory=lambda: _get_default_value("profile_mode", "none"))
    profile_type: str = field(default_factory=lambda: _get_default_value("profile_type", "minimal"))
    nsys_nvtx_include: Optional[List[str]] = field(default_factory=lambda: _get_default_value("nsys_nvtx_include", None))
    enable_nvtx: Optional[bool] = field(default_factory=lambda: _get_default_value("enable_nvtx", None))
    enable_cleanup: bool = field(default_factory=lambda: _get_default_value("enable_cleanup", False))
    use_subprocess: bool = field(default_factory=lambda: _get_default_value("use_subprocess", True))
    execution_mode: Union[ExecutionMode, str, None] = field(default_factory=lambda: _get_default_value("execution_mode", None))
    launch_via: Union[LaunchVia, str] = field(default_factory=lambda: _get_default_value("launch_via", "python"))
    nproc_per_node: Optional[int] = field(default_factory=lambda: _get_default_value("nproc_per_node", None))
    nnodes: Optional[str] = field(default_factory=lambda: _get_default_value("nnodes", None))
    rdzv_backend: Optional[str] = field(default_factory=lambda: _get_default_value("rdzv_backend", None))
    rdzv_endpoint: Optional[str] = field(default_factory=lambda: _get_default_value("rdzv_endpoint", None))
    env_passthrough: List[str] = field(default_factory=lambda: (_get_default_value("env_passthrough", ["CUDA_VISIBLE_DEVICES"]) or []).copy())
    target_extra_args: Dict[str, List[str]] = field(default_factory=lambda: (_get_default_value("target_extra_args", {}) or {}).copy())
    multi_gpu_required: bool = field(default_factory=lambda: bool(_get_default_value("multi_gpu_required", False)))
    target_label: Optional[str] = None
    _execution_mode_overridden: bool = field(init=False, repr=False, default=False)
    
    # Per-stage timeouts (in seconds)
    setup_timeout_seconds: Optional[int] = field(default_factory=lambda: _get_default_value("setup_timeout_seconds", 60))
    warmup_timeout_seconds: Optional[int] = field(default_factory=lambda: _get_default_value("warmup_timeout_seconds", None))
    measurement_timeout_seconds: int = field(default_factory=lambda: _get_default_value("measurement_timeout_seconds", 180))
    enable_gpu_memory_logging: bool = field(default_factory=lambda: _get_default_value("enable_gpu_memory_logging", False))
    gpu_memory_log_interval_seconds: float = field(default_factory=lambda: _get_default_value("gpu_memory_log_interval_seconds", 5.0))
    gpu_memory_log_path: Optional[str] = field(default_factory=lambda: _get_default_value("gpu_memory_log_path", None))
    profiling_timeout_seconds: Optional[int] = field(default_factory=lambda: _get_default_value("profiling_timeout_seconds", None))
    ncu_metric_set: str = field(default_factory=lambda: _get_default_value("ncu_metric_set", "auto"))
    ncu_sampling_interval: int = field(default_factory=lambda: _get_default_value("ncu_sampling_interval", 75000))


    # Legacy timeout field (deprecated, use measurement_timeout_seconds)
    timeout_seconds: int = field(default_factory=lambda: _get_default_value("timeout_seconds", 180))
    
    # Profiler-specific timeouts
    nsys_timeout_seconds: int = field(default_factory=lambda: _get_default_value("nsys_timeout_seconds", 120))
    ncu_timeout_seconds: int = field(default_factory=lambda: _get_default_value("ncu_timeout_seconds", 180))
    proton_timeout_seconds: int = field(default_factory=lambda: _get_default_value("proton_timeout_seconds", 120))
    
    # Timeout multiplier
    timeout_multiplier: float = field(default_factory=lambda: _get_default_value("timeout_multiplier", 1.0))
    
    def __post_init__(self):
        """Set enable_nvtx based on profiling if not explicitly set, apply timeout multiplier, and ensure percentiles is never None."""
        # CRITICAL: Ensure percentiles is always a list, never None (fixes comparison errors)
        if self.percentiles is None:
            self.percentiles = [25, 50, 75, 99]
        if self.env_passthrough is None:
            self.env_passthrough = []
        if self.target_extra_args is None:
            self.target_extra_args = {}
        
        # CRITICAL: Validate and enforce minimum warmup iterations
        # This ensures JIT/compile overhead is NEVER included in measurements.
        # See benchmark/defaults.py for rationale.
        from core.benchmark.defaults import validate_warmup, MINIMUM_WARMUP_ITERATIONS
        self.warmup = validate_warmup(self.warmup, context="BenchmarkConfig")
        
        if self.enable_nvtx is None:
            # Auto-enable NVTX whenever profiling is requested (for nsys traces)
            self.enable_nvtx = self.enable_profiling
        
        # Apply timeout multiplier to all timeout fields
        if self.timeout_multiplier != 1.0:
            def _apply_timeout_multiplier(field_name: str) -> None:
                current_value = getattr(self, field_name)
                if current_value is None:
                    return
                default_value = _get_default_value(field_name, current_value)
                if current_value == default_value:
                    setattr(
                        self,
                        field_name,
                        int(current_value * self.timeout_multiplier),
                    )
            
            _apply_timeout_multiplier("setup_timeout_seconds")
            _apply_timeout_multiplier("warmup_timeout_seconds")
            _apply_timeout_multiplier("measurement_timeout_seconds")
            _apply_timeout_multiplier("profiling_timeout_seconds")
            _apply_timeout_multiplier("nsys_timeout_seconds")
            _apply_timeout_multiplier("ncu_timeout_seconds")
            _apply_timeout_multiplier("proton_timeout_seconds")
            _apply_timeout_multiplier("timeout_seconds")  # Legacy field
        
        # Set defaults for None timeouts
        if self.warmup_timeout_seconds is None:
            # Warmup timeout defaults to measurement timeout
            self.warmup_timeout_seconds = self.measurement_timeout_seconds
        if self.profiling_timeout_seconds is None:
            # Profiling timeout defaults to max of nsys/ncu timeouts
            self.profiling_timeout_seconds = max(self.nsys_timeout_seconds, self.ncu_timeout_seconds, self.proton_timeout_seconds)

        self._execution_mode_overridden = self.execution_mode is not None
        self._sync_execution_mode()
        self._sync_launch_via()

    def _sync_execution_mode(self) -> None:
        """Ensure execution_mode and use_subprocess remain consistent."""
        override = getattr(self, "_execution_mode_overridden", False)
        mode = self.execution_mode
        if not override:
            mode = ExecutionMode.SUBPROCESS if self.use_subprocess else ExecutionMode.THREAD
        else:
            if isinstance(mode, str):
                try:
                    mode = ExecutionMode(mode.lower())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid execution_mode '{mode}'. Expected 'subprocess' or 'thread'."
                    ) from exc
            elif not isinstance(mode, ExecutionMode):
                raise TypeError(
                    f"execution_mode must be a string or ExecutionMode, got {type(mode).__name__}"
                )
        self.use_subprocess = (mode == ExecutionMode.SUBPROCESS)
        self.execution_mode = mode

    def _sync_launch_via(self) -> None:
        """Normalize launch_via to LaunchVia enum."""
        mode = self.launch_via
        if isinstance(mode, str):
            try:
                mode = LaunchVia(mode.lower())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid launch_via '{mode}'. Expected 'python' or 'torchrun'."
                ) from exc
        elif not isinstance(mode, LaunchVia):
            raise TypeError(
                f"launch_via must be a string or LaunchVia, got {type(mode).__name__}"
            )
        self.launch_via = mode

    def set_execution_mode(self, mode: Union[ExecutionMode, str]) -> None:
        """Explicitly set execution mode and mark it as user-overridden."""
        self.execution_mode = mode
        self._execution_mode_overridden = True
        self._sync_execution_mode()
        
        # Backward compatibility: sync timeout_seconds with measurement_timeout_seconds
        # (This ensures legacy code using timeout_seconds still works)
        if self.timeout_seconds == 15 and self.measurement_timeout_seconds != 15:
            # If timeout_seconds wasn't explicitly changed, sync it
            self.timeout_seconds = self.measurement_timeout_seconds
    
    def get_effective_timeout(self, stage: str) -> Optional[int]:
        """Get effective timeout for a given stage, accounting for multiplier.
        
        Args:
            stage: One of 'setup', 'warmup', 'measurement', 'profiling', 'nsys', 'ncu', 'proton'
            
        Returns:
            Timeout in seconds, or None if no timeout for this stage
        """
        timeouts = {
            'setup': self.setup_timeout_seconds,
            'warmup': self.warmup_timeout_seconds,
            'measurement': self.measurement_timeout_seconds,
            'profiling': self.profiling_timeout_seconds,
            'nsys': self.nsys_timeout_seconds,
            'ncu': self.ncu_timeout_seconds,
            'proton': self.proton_timeout_seconds,
        }
        return timeouts.get(stage)


@dataclass
class WorkloadMetadata:
    """Metadata describing workload units processed per benchmark iteration."""
    
    requests_per_iteration: float = 1.0
    tokens_per_iteration: Optional[float] = None
    samples_per_iteration: Optional[float] = None
    bytes_per_iteration: Optional[float] = None
    custom_units_per_iteration: Optional[float] = None
    custom_unit_name: Optional[str] = None
    goodput: Optional[float] = None


@dataclass
class TorchrunLaunchSpec:
    """Declarative description of a torchrun invocation."""

    script_path: Path
    script_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    parse_rank0_only: bool = True
    multi_gpu_required: bool = False
    name: Optional[str] = None
    config_arg_map: Dict[str, str] = field(default_factory=dict)


# BenchmarkResult is provided by benchmark_models.py
BenchmarkResult = PydanticBenchmarkResult


class BaseBenchmark:
    """Base class for benchmarks with shared functionality.
    
    Provides common patterns for device resolution, setup, teardown, validation,
    NVTX range management (_nvtx_range()), and CUDA synchronization (_synchronize()).
    Benchmarks should inherit from this class (or a helper base that derives from it).
    
    Usage:
        class MyBenchmark(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.model = None
                self.data = None
            
            def setup(self) -> None:
                torch.manual_seed(42)
                self.model = nn.Linear(256, 256).to(self.device)
                self.data = torch.randn(32, 256, device=self.device)
                torch.cuda.synchronize()
            
            def benchmark_fn(self) -> None:
                with self._nvtx_range("my_benchmark"):
                    _ = self.model(self.data)
    """
    
    def __init__(self):
        """Initialize benchmark with device resolution.
        
        Subclasses should call super().__init__() and then set up their own attributes.
        """
        self.device = self._resolve_device()
        self._config = None  # Cache for get_config()
        self._workload_metadata: Optional[WorkloadMetadata] = None
    
    def _resolve_device(self) -> torch.device:
        """Resolve CUDA device, failing fast if CUDA is not available.
        
        Returns:
            torch.device("cuda") if CUDA is available
            
        Raises:
            RuntimeError: If CUDA is not available (NVIDIA GPU required)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required - NVIDIA GPU and tools must be available")
        return torch.device("cuda")
    
    def setup(self) -> None:
        """Setup phase: initialize models, data, etc.
        
        Subclasses should override this method to implement their specific setup logic.
        """
        pass
    
    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args.
        
        Subclasses must override this method to implement their benchmark logic.
        """
        raise NotImplementedError("Subclasses must implement benchmark_fn()")
    
    def teardown(self) -> None:
        """Cleanup phase.
        
        Default implementation clears CUDA cache. Subclasses can override
        to add additional cleanup logic.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clear any cached config from a previous harness run
        self._config = None
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Return the active harness config when available.
        
        Subclasses can override to provide custom overrides; the harness
        will still stash the merged config on ``self._config`` for runtime checks.
        """
        return getattr(self, "_config", None)

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> Optional[TorchrunLaunchSpec]:
        """Optional torchrun launch description for multi-GPU benchmarks."""
        return None
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result, return error message if invalid.
        
        Default implementation returns None (no validation).
        Subclasses should override to implement validation logic.
        
        Returns:
            None if validation passes, or error message string if validation fails
        """
        return None

    def skip_input_verification(self) -> bool:
        """Return True to skip input equivalence verification for this benchmark.
        
        Override this in benchmarks where input verification is not applicable
        (e.g., benchmarks that intentionally use different input sizes to test
        scaling behavior, or informational benchmarks).
        
        Returns:
            False by default (verification enabled)
        """
        # Allow opt-in skips via attribute without overriding the method
        if getattr(self, "skip_input_check", False):
            return True
        return False

    def skip_output_verification(self) -> bool:
        """Return True to skip output correctness verification for this benchmark.
        
        Override this in benchmarks where output verification is not applicable
        (e.g., benchmarks that produce non-deterministic outputs, or benchmarks
        that test different algorithms with legitimately different results).
        
        Returns:
            False by default (verification enabled)
        """
        # Allow opt-in skips via attribute without overriding the method
        if getattr(self, "skip_output_check", False):
            return True
        return False

    def get_input_signature(self) -> Optional[Dict[str, Any]]:
        """Return a signature describing this benchmark's input workload.
        
        Used by the harness to verify that baseline and optimized benchmarks
        operate on equivalent workloads. Without this verification, performance
        comparisons are meaningless.
        
        The signature should capture all parameters that affect workload size:
        - Tensor shapes (batch_size, seq_len, hidden_size, etc.)
        - Model parameters (num_layers, num_heads, etc.)
        - Any other configuration that affects computational work
        
        Default implementation infers from common attributes (both instance
        and class-level). Override for custom workload descriptions.
        
        Returns:
            Dict describing the input workload, or None if inference fails
        """
        signature: Dict[str, Any] = {}
        
        # Common workload attributes to capture (lowercase instance attrs)
        workload_attrs = [
            # Tensor dimensions
            "batch_size", "B", "seq_len", "seq_length", "sequence_length",
            "hidden_size", "hidden_dim", "d_model", "embed_dim",
            "num_heads", "n_heads", "nheads",
            "num_layers", "n_layers", "nlayers",
            "vocab_size", "num_classes",
            # Matrix dimensions
            "M", "N", "K", "m", "n", "k",
            # Image dimensions
            "height", "width", "channels", "H", "W", "C",
            # Other common params
            "num_tokens", "num_samples", "num_elements",
            "input_size", "output_size",
            # MoE-specific
            "num_experts", "num_experts_per_tok", "intermediate_size",
        ]
        
        # Also check class-level UPPERCASE constants (common pattern)
        class_attrs = [
            "BATCH_SIZE", "SEQ_LEN", "SEQ_LENGTH", "SEQUENCE_LENGTH",
            "HIDDEN_SIZE", "HIDDEN_DIM", "D_MODEL", "EMBED_DIM",
            "NUM_HEADS", "N_HEADS", "NHEADS",
            "NUM_LAYERS", "N_LAYERS", "NLAYERS",
            "VOCAB_SIZE", "NUM_CLASSES",
            "NUM_EXPERTS", "NUM_EXPERTS_PER_TOK", "INTERMEDIATE_SIZE",
            "WIDTH", "HEIGHT", "CHANNELS",
        ]
        
        for attr in workload_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    signature[attr] = value
        
        # Check class-level constants
        for attr in class_attrs:
            if hasattr(type(self), attr):
                value = getattr(type(self), attr)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # Use lowercase key for consistency
                    signature[attr.lower()] = value
        
        # Note: We intentionally do NOT capture tensor shapes/dtypes here.
        # Tensors are typically created in setup() which runs after signature comparison.
        # The workload attributes above (batch_size, seq_len, etc.) are sufficient
        # to verify that benchmarks have equivalent workloads.
        
        return signature if signature else None

    def _verify_kernel(
        self,
        kernel_fn,
        reference_fn,
        shape: tuple,
        *,
        formal: bool = False,
        kernel_source: Optional[str] = None,
    ) -> tuple:
        """Verify kernel correctness using core/verification/kernel_correctness.
        
        Helper method for benchmarks to verify their kernels match a reference.
        Can be called in validate_result() or anywhere else.
        
        Args:
            kernel_fn: The kernel function to verify
            reference_fn: Reference implementation to compare against
            shape: Input tensor shape for testing
            formal: If True, use formal verification with proofs (slower but thorough)
            kernel_source: CUDA source code (required if formal=True)
            
        Returns:
            Tuple of (passed: bool, result: VerificationResult or FormalVerificationResult)
            
        Example:
            def validate_result(self) -> Optional[str]:
                passed, result = self._verify_kernel(
                    self.my_kernel, 
                    self.reference_impl,
                    shape=(1024, 1024),
                )
                if not passed:
                    return f"Kernel verification failed: {result}"
                return None
        """
        try:
            if formal:
                from core.verification.kernel_correctness import FormalKernelVerifier
                verifier = FormalKernelVerifier(device=str(self.device))
                result = verifier.verify(kernel_fn, reference_fn, kernel_source or "")
                return (result.all_proven, result)
            else:
                from core.verification.kernel_correctness import ManualKernelVerifier
                verifier = ManualKernelVerifier(device=str(self.device))
                result = verifier.verify(kernel_fn, reference_fn, shape)
                return (result.all_passed, result)
        except ImportError:
            # Verification tools not available
            return (True, None)

    def register_workload_metadata(
        self,
        *,
        requests_per_iteration: Optional[float] = None,
        tokens_per_iteration: Optional[float] = None,
        samples_per_iteration: Optional[float] = None,
        bytes_per_iteration: Optional[float] = None,
        custom_units_per_iteration: Optional[float] = None,
        custom_unit_name: Optional[str] = None,
        goodput: Optional[float] = None,
    ) -> None:
        """Register workload metadata for throughput calculations."""
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=requests_per_iteration if requests_per_iteration is not None else 1.0,
            tokens_per_iteration=tokens_per_iteration,
            samples_per_iteration=samples_per_iteration,
            bytes_per_iteration=bytes_per_iteration,
            custom_units_per_iteration=custom_units_per_iteration,
            custom_unit_name=custom_unit_name,
            goodput=goodput,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Return workload metadata if registered."""
        if self._workload_metadata is not None:
            return self._workload_metadata
        inferred = self._infer_workload_metadata()
        if inferred is not None:
            self._workload_metadata = inferred
        return self._workload_metadata

    def _infer_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Best-effort inference for benchmarks that rely on external configs.
        
        This inspects common attribute names (batch_size, seq_len, hidden_size, etc.)
        to synthesize conservative workload metadata so downstream tools can make
        apples-to-apples comparisons even when individual benchmarks omit explicit
        WorkloadMetadata registration. The goal is cosmetic consistency for review
        tooling; benchmarks can still override via register_workload_metadata().
        """

        def _collect_numeric_attrs() -> Dict[str, float]:
            values: Dict[str, float] = {}
            for attr in dir(self):
                if attr.startswith("_"):
                    continue
                try:
                    value = getattr(self, attr)
                except Exception:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    values[attr] = float(value)
            return values

        numeric_attrs = _collect_numeric_attrs()
        if not numeric_attrs:
            return None

        def _pick(*names: str) -> Optional[float]:
            for name in names:
                if name in numeric_attrs:
                    return numeric_attrs[name]
            return None

        requests = _pick(
            "requests_per_iteration",
            "num_requests",
            "request_count",
            "requests",
            "batches_per_iteration",
            "microbatch_size",
            "micro_batch_size",
            "batch_size",
            "B",
        )

        tokens = _pick(
            "tokens_per_iteration",
            "token_count",
            "tokens",
            "num_tokens",
            "total_tokens",
            "elements_per_iteration",
        )

        def _product(names: Tuple[str, ...]) -> Optional[float]:
            prod = 1.0
            found = False
            for name in names:
                value = numeric_attrs.get(name)
                if value is None:
                    return None
                prod *= value
                found = True
            return prod if found else None

        if tokens is None:
            combo_candidates = [
                ("batch_size", "tokens_per_step", "decode_steps"),
                ("batch_size", "sequence_length"),
                ("batch_size", "seq_len"),
                ("batch_size", "hidden_size"),
                ("microbatch_size", "hidden_size"),
                ("micro_batch_size", "hidden_size"),
                ("batch_size", "input_len"),
                ("batch_size", "model_dim"),
                ("batch_size", "embedding_dim"),
                ("N", "N"),
                ("N", "K"),
                ("M", "N"),
                ("rows", "cols"),
                ("size", "size"),
            ]
            for combo in combo_candidates:
                prod = _product(combo)
                if prod is not None and prod > 0:
                    tokens = prod
                    break

        bytes_per_iter = _pick(
            "bytes_per_iteration",
            "byte_count",
            "total_bytes",
        )

        if tokens is None and bytes_per_iter is None and requests is None:
            return None

        return WorkloadMetadata(
            requests_per_iteration=requests if requests is not None else 1.0,
            tokens_per_iteration=tokens,
            bytes_per_iteration=bytes_per_iter,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Benchmarks can override to expose custom metrics."""
        return None

    def get_optimization_goal(self) -> str:
        """Return the primary optimization goal for this benchmark.
        
        Override this method to indicate what metric the optimization targets.
        The harness will use this to evaluate success appropriately.
        
        Returns:
            One of: "speed", "memory", "throughput", "latency", "power"
            Default is "speed" (lower time = better)
            
        Examples:
            - "speed": Traditional speedup (baseline_time / optimized_time)
            - "memory": Memory reduction (baseline_memory / optimized_memory)
            - "throughput": Higher is better (optimized_throughput / baseline_throughput)
            - "latency": Lower is better (baseline_latency / optimized_latency)
        """
        return "speed"
    
    def _record_start(self) -> float:
        """Return a high-resolution timestamp for chunk-level timing."""
        return time.perf_counter()

    def _record_stop(self, start: float) -> float:
        """Return elapsed time in milliseconds since ``start``."""
        return (time.perf_counter() - start) * 1000.0
    
    def _scale_workload_by_memory(self, base_size: int) -> int:
        """Scale workload size based on available GPU memory.
        
        Args:
            base_size: Base workload size for large GPUs (>=16GB)
            
        Returns:
            Scaled workload size based on GPU memory:
            - >=16GB: base_size (100%)
            - >=8GB: base_size * 0.5 (50%)
            - >=4GB: base_size * 0.25 (25%)
            - <4GB: base_size * 0.1 (10%)
        """
        if not torch.cuda.is_available():
            return base_size
        
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_memory_gb >= 16:
            return base_size
        elif total_memory_gb >= 8:
            return int(base_size * 0.5)
        elif total_memory_gb >= 4:
            return int(base_size * 0.25)
        else:
            return int(base_size * 0.1)

    def to_device(
        self,
        tensor: torch.Tensor,
        *,
        non_blocking: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Copy ``tensor`` to ``device`` while defaulting to non_blocking=True when the
        source resides in pinned host memory. Benchmarks can override ``device`` or
        ``non_blocking`` for synchronous copies.
        """

        target = device if device is not None else getattr(self, "device", None)
        if target is None:
            target = self._resolve_device()
        if target.type == "cuda" and hasattr(tensor, "is_pinned") and not tensor.is_cuda:
            blocking = tensor.is_pinned() if non_blocking is None else non_blocking
            return tensor.to(target, non_blocking=bool(blocking))
        return tensor.to(target)
    
    @contextmanager
    def _nvtx_range(self, name: str):
        """Context manager for NVTX ranges with automatic enable/disable.
        
        Automatically checks if NVTX is enabled via get_config().
        
        Args:
            name: Name for the NVTX range
            
        Usage:
            with self._nvtx_range("my_operation"):
                # code to profile
                pass
        """
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        # Prefer the harness-merged config if present to honor CLI flags
        config = getattr(self, "_config", None) or self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range(name, enable=enable_nvtx):
            yield
    
    def _synchronize(self) -> None:
        """Synchronize CUDA device if available.
        
        Convenience method for benchmarks to ensure operations complete.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)


class BenchmarkHarness:
    """Production-grade benchmarking harness with profiling support."""
    
    def __init__(
        self,
        mode: BenchmarkMode = BenchmarkMode.CUSTOM,
        config: Optional[BenchmarkConfig] = None
    ):
        self.mode = mode
        self.config = config or BenchmarkConfig()
        self.config._sync_execution_mode()
        self.config._sync_launch_via()
        self.device = self.config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._seed_info = self._setup_reproducibility()  # Store seed info for manifest
        self._thread_executor: Optional[ThreadPoolExecutor] = None
    
    def _setup_reproducibility(self) -> Dict[str, Any]:
        """Setup for reproducible benchmarks.
        
        Returns:
            Dictionary with seed values that were set (for manifest capture).
        """
        seed_info: Dict[str, Any] = {
            "random_seed": None,
            "numpy_seed": None,
            "torch_seed": None,
            "cuda_seed": None,
            "deterministic_mode": self.config.deterministic,
        }
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        if self.config.seed is not None:
            seed = self.config.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            seed_info["random_seed"] = seed
            seed_info["numpy_seed"] = seed
            seed_info["torch_seed"] = seed
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                seed_info["cuda_seed"] = seed
            
            if LOGGER_AVAILABLE:
                logger.info(f"Seeds set to {seed} (random, numpy, torch, cuda)")
        
        # Log deterministic mode warning if enabled
        if self.config.deterministic and LOGGER_AVAILABLE:
            logger.info(
                "Deterministic mode enabled (may impact performance by 5-20%). "
                "This ensures bitwise reproducibility, but forces slower fallback kernels and ops without deterministic support may raise."
            )
        
        return seed_info

    @contextmanager
    def _nvtx_range(self, name: str):
        """Context manager for NVTX ranges with automatic enable/disable.
        
        Args:
            name: Name for the NVTX range
        """
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.config
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range(name, enable=enable_nvtx):
            yield

    def _world_size_hint(self, config: BenchmarkConfig) -> Optional[int]:
        """Best-effort world size estimation from config."""
        nproc = getattr(config, "nproc_per_node", None)
        nnodes = getattr(config, "nnodes", None)
        world_size: Optional[int] = None
        try:
            if nproc is not None:
                world_size = int(nproc)
        except Exception:
            world_size = None
        if nnodes is not None:
            try:
                nodes_int = int(str(nnodes).split(":")[0])
                world_size = (world_size or 1) * nodes_int
            except ValueError:
                pass  # nnodes not a valid integer
        if world_size is None:
            if getattr(config, "launch_via", LaunchVia.PYTHON) == LaunchVia.PYTHON:
                world_size = 1
            elif torch.cuda.is_available():
                world_size = torch.cuda.device_count()
        return world_size

    def _annotate_launch_metadata(
        self,
        result: PydanticBenchmarkResult,
        config: BenchmarkConfig,
        *,
        world_size: Optional[int] = None,
        multi_gpu_required: Optional[bool] = None,
    ) -> None:
        """Attach launch metadata onto a BenchmarkResult in-place."""
        result.launch_via = getattr(config, "launch_via", None)
        resolved_world_size = world_size if world_size is not None else self._world_size_hint(config)
        result.world_size = resolved_world_size
        requires_multi = getattr(config, "multi_gpu_required", False) if multi_gpu_required is None else multi_gpu_required
        result.multi_gpu_required = bool(requires_multi)
        result.multi_gpu = bool(resolved_world_size and resolved_world_size > 1)

    def _apply_target_overrides(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> None:
        """Apply per-target CLI overrides to a benchmark instance, if supported."""
        target_label = getattr(config, "target_label", None)
        if not target_label:
            return
        overrides_map = getattr(config, "target_extra_args", {}) or {}
        overrides = overrides_map.get(target_label)
        if not overrides:
            return
        argv = shlex.split(overrides) if isinstance(overrides, str) else list(overrides)
        if not argv:
            return
        hook = getattr(benchmark, "apply_target_overrides", None)
        if callable(hook):
            try:
                hook(argv)
            except Exception:
                if LOGGER_AVAILABLE:
                    logger.warning("Ignoring target overrides for %s due to error", target_label, exc_info=True)
    
    def _create_timeout_result(
        self,
        *,
        stage: str,
        duration: float,
        limit: Optional[float],
        errors: List[str],
        benchmark_name: str,
        config: BenchmarkConfig,
        watchdog: Optional[Dict[str, Any]] = None,
    ) -> PydanticBenchmarkResult:
        """Helper to build a timeout result with consistent schema."""
        timeout_timing = TimingStats(
            mean_ms=0.0,
            median_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            p50_ms=None,
            p90_ms=None,
            p95_ms=None,
            p99_ms=None,
            percentiles={},
            iterations=0,
            warmup_iterations=config.warmup,
            raw_times_ms=[],
            schemaVersion="1.0",
        )
        result = PydanticBenchmarkResult(
            timing=timeout_timing,
            errors=errors,
            timeout_stage=stage,
            timeout_duration_seconds=duration,
            timeout_limit_seconds=int(limit) if limit is not None else None,
            inference_timing=None,
            memory=None,
            artifacts=None,
            profiler_metrics=None,
            validation_status=None,
            validation_message=json.dumps(watchdog) if watchdog else None,
            benchmark_name=benchmark_name,
            device=str(self.device),
            mode=self.mode.value,
            schemaVersion="1.0",
        )
        self._annotate_launch_metadata(result, config, world_size=self._world_size_hint(config))
        return result
    
    def _benchmark_with_torchrun(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> PydanticBenchmarkResult:
        """Launch benchmark via torchrun for multi-GPU targets."""
        import inspect
        print("[harness] _benchmark_with_torchrun start", flush=True)

        def _filter_logs(lines: List[str], dedupe: bool = True) -> List[str]:
            filtered: List[str] = []
            seen: set = set()
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                key = stripped.lower()
                if dedupe and key in seen:
                    continue
                seen.add(key)
                # Keep messages that either don't mention rank or explicitly mention rank 0
                if "rank 0" in key or "local_rank: 0" in key or "r0" in key:
                    filtered.append(stripped)
                elif "rank" not in key or "world_size" in key:
                    filtered.append(stripped)
            return filtered

        def _extract_tokens_per_s(lines: List[str]) -> Optional[float]:
            pattern = re.compile(r"([0-9][0-9,\\.]+)\\s*(tok(?:ens)?/s|toks/s|tok/s)", re.IGNORECASE)
            best: Optional[float] = None
            for line in lines:
                match = pattern.search(line)
                if not match:
                    continue
                try:
                    candidate = float(match.group(1).replace(",", ""))
                    best = candidate if best is None else max(best, candidate)
                except ValueError:
                    continue
            return best

        errors: List[str] = []
        spec: Optional[TorchrunLaunchSpec] = None
        try:
            getter = getattr(benchmark, "get_torchrun_spec", None)
            if callable(getter):
                spec = getter(config)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Failed to build torchrun spec: {exc}")
        if spec is None:
            # Fallback to module path if provided
            benchmark_module = inspect.getmodule(benchmark) or inspect.getmodule(benchmark.__class__)
            module_path = Path(getattr(benchmark_module, "__file__", "")).resolve()
            spec = TorchrunLaunchSpec(script_path=module_path)
        print(f"[harness] torchrun spec script={spec.script_path} args={spec.script_args}", flush=True)

        if not spec or not Path(spec.script_path).exists():
            raise RuntimeError("SKIPPED: Torchrun launch requested but no valid script_path was provided.")

        world_size_hint = self._world_size_hint(config)
        multi_gpu_required = bool(spec.multi_gpu_required or getattr(config, "multi_gpu_required", False))
        if multi_gpu_required and (world_size_hint is not None and world_size_hint < 2):
            message = "SKIPPED: Multi-GPU benchmark requires nproc_per_node>=2."
            errors.append(message)
            raise RuntimeError(message)

        nproc_per_node = getattr(config, "nproc_per_node", None) or 1
        torchrun_cmd: List[str] = [
            "torchrun",
            "--nproc_per_node",
            str(nproc_per_node),
        ]
        if getattr(config, "nnodes", None):
            torchrun_cmd.extend(["--nnodes", str(config.nnodes)])
        if getattr(config, "rdzv_backend", None) or getattr(config, "nnodes", None):
            torchrun_cmd.extend(["--rdzv_backend", getattr(config, "rdzv_backend", None) or "c10d"])
        if getattr(config, "rdzv_endpoint", None):
            torchrun_cmd.extend(["--rdzv_endpoint", str(config.rdzv_endpoint)])

        def _config_args_from_map() -> List[str]:
            args: List[str] = []
            for key, flag in spec.config_arg_map.items():
                if not flag or not hasattr(config, key):
                    continue
                value = getattr(config, key)
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        args.append(flag)
                else:
                    args.extend([flag, str(value)])
            return args

        extra_args: List[str] = []
        target_label = getattr(config, "target_label", None)
        if target_label:
            target_overrides = (getattr(config, "target_extra_args", {}) or {}).get(target_label)
            if target_overrides:
                if isinstance(target_overrides, str):
                    extra_args.extend(shlex.split(target_overrides))
                else:
                    extra_args.extend(list(target_overrides))

        script_args = list(spec.script_args)
        script_args.extend(_config_args_from_map())
        script_args.extend(extra_args)

        full_cmd = torchrun_cmd + [str(spec.script_path)] + script_args
        print(f"[harness] torchrun cmd: {' '.join(full_cmd)}", flush=True)

        env = os.environ.copy()
        for key in getattr(config, "env_passthrough", []) or []:
            if key in os.environ:
                env[key] = os.environ[key]
        print(f"[harness] torchrun env passthrough: {getattr(config, 'env_passthrough', [])}", flush=True)
        env.update(spec.env)

        stdout = ""
        stderr = ""
        elapsed = 0.0
        timeout_limit = config.get_effective_timeout("measurement")

        try:
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
                env=env,
            )
            start = time.time()
            try:
                stdout, stderr = process.communicate(timeout=timeout_limit)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                elapsed = time.time() - start
                return self._create_timeout_result(
                    stage="measurement",
                    duration=elapsed,
                    limit=timeout_limit,
                    errors=["Torchrun launch exceeded timeout", *errors],
                    benchmark_name=spec.name or getattr(benchmark, "name", None) or benchmark.__class__.__name__,
                    config=config,
                )
            elapsed = time.time() - start
            if process.returncode != 0:
                preview = stderr.strip().splitlines()[-5:] if stderr else []
                errors.append(f"torchrun exited with code {process.returncode}")
                if preview:
                    errors.append("stderr: " + " | ".join(preview))
        except FileNotFoundError:
            raise RuntimeError("SKIPPED: torchrun not found in PATH.")

        stdout_lines = stdout.splitlines() if stdout else []
        filtered_lines = _filter_logs(stdout_lines, dedupe=spec.parse_rank0_only)
        tokens_per_s = _extract_tokens_per_s(filtered_lines)

        iterations = getattr(config, "iterations", None) or 1
        per_iter_ms = (elapsed * 1000.0) / max(iterations, 1)
        times_ms = [per_iter_ms] * max(iterations, 1)
        result = self._compute_stats(times_ms, config)

        if tokens_per_s is not None:
            result.throughput = ThroughputStats(
                tokens_per_s=tokens_per_s,
                requests_per_s=None,
                samples_per_s=None,
                bytes_per_s=None,
                custom_unit_per_s=None,
                custom_unit_name=None,
                latency_ms=result.timing.mean_ms,
                goodput=None,
                schemaVersion="1.0",
            )
            result.custom_metrics["tokens_per_s"] = tokens_per_s

        if filtered_lines:
            tail = "\n".join(filtered_lines[-5:])
            result.validation_message = tail

        if errors:
            result.errors.extend(errors)
        if stderr and not errors:
            result.errors.append(stderr.strip())

        self._annotate_launch_metadata(
            result,
            config,
            world_size=world_size_hint,
            multi_gpu_required=multi_gpu_required,
        )
        return result
    
    def _ensure_thread_executor(self) -> None:
        """Initialize thread executor for threaded execution mode."""
        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="benchmark-thread",
            )
    
    def _reset_thread_executor(self) -> None:
        """Shutdown and reset thread executor."""
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=False, cancel_futures=True)
            self._thread_executor = None
    
    @contextmanager
    def _memory_tracking(self, config: Optional[BenchmarkConfig] = None):
        """Context manager for memory tracking.
        
        Yields a MemoryStats pydantic model (or None if tracking disabled).
        
        Args:
            config: BenchmarkConfig to check for enable_memory_tracking. If None, uses self.config.
        
        Yields:
            MemoryStats pydantic model if tracking enabled, None otherwise.
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for memory tracking. Install with: pip install pydantic")
        
        # Use provided config or fall back to instance config
        check_config = config if config is not None else self.config
        
        if not check_config.enable_memory_tracking or not torch.cuda.is_available():
            yield None
            return
        
        # Create MemoryStats model
        result = MemoryStats(peak_mb=None, allocated_mb=None, reserved_mb=None, schemaVersion="1.0")
        
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        yield result
        torch.cuda.synchronize(self.device)
        
        # Set memory values
        result.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        result.allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
        
        # Try to get reserved memory if available
        try:
            result.reserved_mb = torch.cuda.memory_reserved(self.device) / (1024**2)
        except AttributeError:
            # memory_reserved may not be available in all PyTorch versions
            result.reserved_mb = None
    
    def benchmark(
        self,
        benchmark: Union[BaseBenchmark, Callable[[], Any]],
        name: Optional[str] = None
    ) -> PydanticBenchmarkResult:
        """Run benchmark and return statistical results.
        
        Uses subprocess isolation (if enabled) or threading timeout to prevent hangs.
        Default timeout is 15 seconds.
        """
        callable_wrapped = False
        # Support callable benchmarks by wrapping in a minimal BaseBenchmark
        if not isinstance(benchmark, BaseBenchmark):
            fn = benchmark
            callable_wrapped = True
            
            class _CallableBenchmark(BaseBenchmark):  # type: ignore[misc]
                def __init__(self, wrapped_fn: Callable[[], Any], bench_name: Optional[str]):
                    super().__init__()
                    self._fn = wrapped_fn
                    self.name = bench_name or "callable_benchmark"
                
                def setup(self) -> None:
                    # No-op setup by default for raw callables
                    pass
                
                def benchmark_fn(self) -> None:
                    self._fn()
            
            benchmark = _CallableBenchmark(fn, name)
        elif name and getattr(benchmark, "name", None) is None:
            # Preserve provided name if the benchmark did not set one
            benchmark.name = name
        
        print("[harness] benchmark() start", flush=True)
        # Clone config to avoid mutating shared instance; deepcopy prevents
        # dataclasses.replace from re-running __post_init__ (which re-applies multipliers)
        config = copy.deepcopy(self.config)
        print(f"[harness] initial launch_via={config.launch_via} execution_mode={config.execution_mode}", flush=True)
        if callable_wrapped:
            # Callable benchmarks run in-process to avoid JSON parsing of custom stdout
            config.use_subprocess = False
            config.execution_mode = ExecutionMode.THREAD
        bench_config = benchmark.get_config()
        if bench_config and _is_chapter_or_labs_benchmark(benchmark):
            bench_config.iterations = None  # type: ignore[assignment]
            bench_config.warmup = None  # type: ignore[assignment]
        if bench_config:
            # Override with benchmark-specific settings
            for key, value in bench_config.__dict__.items():
                if value is not None:
                    # Never allow benchmarks to re-enable subprocess mode when we're already
                    # running inside an isolated subprocess (prevents recursive launches).
                    if key in ("use_subprocess", "execution_mode") and self.config.use_subprocess is False:
                        continue
                    if key == "target_extra_args":
                        if value:
                            config.target_extra_args = {
                                **(getattr(config, "target_extra_args", {}) or {}),
                                **value,
                            }
                        continue
                    if key == "launch_via":
                        default_launch = _get_default_value("launch_via", None)
                        def _norm(val):
                            if val is None:
                                return None
                            if hasattr(val, "value"):
                                return str(getattr(val, "value")).lower()
                            return str(val).lower()
                        if (
                            _norm(config.launch_via) is not None
                            and _norm(default_launch) is not None
                            and _norm(config.launch_via) != _norm(default_launch)
                            and _norm(value) == _norm(default_launch)
                        ):
                            # Preserve caller-specified launcher when benchmark config only supplies the default
                            continue
                    if key == "env_passthrough" and not value:
                        continue
                    setattr(config, key, value)
        config._sync_execution_mode()
        config._sync_launch_via()
        print(f"[harness] execution_mode={config.execution_mode} launch_via={config.launch_via}", flush=True)
        
        previous_config = getattr(benchmark, "_config", None)
        # Make merged config visible to benchmarks that need per-target args.
        benchmark._config = config  # type: ignore[attr-defined]
        
        # CRITICAL: Ensure percentiles is always a list (never None)
        # This handles cases where benchmark.get_config() returns a config with percentiles=None
        # or where the config was created before __post_init__ could fix it
        if config.percentiles is None or not isinstance(config.percentiles, list):
            config.percentiles = [25, 50, 75, 99]
        
        gpu_mem_logger: Optional[GpuMemoryLogger] = None
        # GPU memory logging is disabled for smoke stability.
        if False and should_enable_gpu_memory_logging(getattr(config, "enable_gpu_memory_logging", False)):
            log_interval = resolve_gpu_log_interval(getattr(config, "gpu_memory_log_interval_seconds", 5.0))
            log_path = resolve_gpu_log_path(getattr(config, "gpu_memory_log_path", None))
            gpu_mem_logger = GpuMemoryLogger(self.device, interval=log_interval, log_path=log_path)
            if gpu_mem_logger.start():
                if LOGGER_AVAILABLE:
                    logger.info(
                        "GPU memory logging enabled: %s (interval=%.2fs)",
                        log_path,
                        log_interval,
                    )
            else:
                gpu_mem_logger = None

        try:
            if config.launch_via == LaunchVia.TORCHRUN:
                print("[harness] dispatch torchrun", flush=True)
                return self._benchmark_with_torchrun(benchmark, config)
            if config.execution_mode == ExecutionMode.SUBPROCESS:
                print("[harness] dispatch subprocess", flush=True)
                return self._benchmark_with_subprocess(benchmark, config)
            print("[harness] dispatch threading (direct)", flush=True)
            return self._benchmark_with_threading(benchmark, config)
        finally:
            benchmark._config = previous_config  # type: ignore[attr-defined]
            if gpu_mem_logger is not None:
                log_file = gpu_mem_logger.stop()
                if LOGGER_AVAILABLE:
                    logger.info("GPU memory log saved to %s", log_file)
    
    def benchmark_with_manifest(
        self, 
        benchmark: BaseBenchmark,
        run_id: Optional[str] = None
    ) -> BenchmarkRun:
        """Run benchmark with manifest and return BenchmarkRun.
        
        This method creates a RunManifest at the start, runs the benchmark,
        and returns a BenchmarkRun containing both the manifest and result.
        
        Args:
            benchmark: BaseBenchmark instance to run
            run_id: Optional run identifier (defaults to timestamp)
        
        Returns:
            BenchmarkRun with manifest and result
        """
        if not PYDANTIC_AVAILABLE or BenchmarkRun is None:
            raise ImportError("pydantic and BenchmarkRun are required for benchmark_with_manifest")
        
        from datetime import datetime
        from core.benchmark.run_manifest import RunManifest
        
        # Create manifest at start
        start_time = datetime.now()
        config_dict = self.config.__dict__.copy()
        manifest = RunManifest.create(config=config_dict, start_time=start_time)
        
        # Add seed information to manifest
        from core.benchmark.run_manifest import SeedInfo
        if self._seed_info:
            manifest.seeds = SeedInfo(
                random_seed=self._seed_info.get("random_seed"),
                numpy_seed=self._seed_info.get("numpy_seed"),
                torch_seed=self._seed_info.get("torch_seed"),
                cuda_seed=self._seed_info.get("cuda_seed"),
                deterministic_mode=bool(self._seed_info.get("deterministic_mode", False)),
                schemaVersion="1.0",
            )
        
        # Run benchmark
        result = self.benchmark(benchmark)
        
        # Finalize manifest
        end_time = datetime.now()
        manifest.finalize(end_time=end_time)
        
        # Generate run_id if not provided
        if run_id is None:
            run_id = start_time.strftime("run_%Y%m%d_%H%M%S")
        
        # Create BenchmarkRun
        return BenchmarkRun(
            manifest=manifest,
            result=result,
            run_id=run_id,
            timestamp=start_time.isoformat(),
            schemaVersion="1.0",
        )
    
    def _benchmark_with_subprocess(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> PydanticBenchmarkResult:
        """Run benchmark in subprocess for reliable timeout cancellation."""
        import json
        import inspect
        
        errors: List[str] = []
        memory_peak_mb: Optional[float] = None
        memory_allocated_mb: Optional[float] = None
        profiling_outputs: Dict[str, str] = {}
        nsys_metrics: Dict[str, float] = {}
        ncu_metrics: Dict[str, float] = {}
        proton_metrics: Dict[str, float] = {}
        times_ms: List[float] = []
        inference_timing_data: Optional[Dict[str, List[float]]] = None
        
        # Get benchmark name and module info for error messages
        module_override = getattr(benchmark, "_module_file_override", None)
        class_override = getattr(benchmark, "_factory_name_override", None)
        benchmark_module = inspect.getmodule(benchmark)
        benchmark_class = class_override or benchmark.__class__.__name__
        benchmark_name = getattr(benchmark, '__name__', None) or getattr(benchmark, 'name', None) or benchmark_class
        
        if benchmark_module is None:
            benchmark_module = inspect.getmodule(benchmark.__class__)
        
        if benchmark_module is None:
            # Fallback to threading if we can't determine module
            return self._benchmark_with_threading(benchmark, config)
        
        module_file = module_override or getattr(benchmark_module, "__file__", None)
        if module_file is None:
            spec = getattr(benchmark_module, "__spec__", None)
            if spec is not None:
                module_file = getattr(spec, "origin", None)
        
        if module_file is None:
            # Fallback to threading if we can't determine module file
            return self._benchmark_with_threading(benchmark, config)
        
        module_path = Path(module_file).resolve()
        if not module_path.exists():
            # Fallback to threading if module file doesn't exist
            return self._benchmark_with_threading(benchmark, config)
        
        # Prepare config dict (serialize only simple types)
        config_dict = {}
        for key in ['iterations', 'warmup', 'min_run_time_ms', 'percentiles', 'enable_memory_tracking',
                   'deterministic', 'seed', 'enable_profiling', 'enable_nsys', 'enable_ncu', 'enable_proton',
                   'profiling_output_dir', 'enable_nvtx', 'enable_cleanup', 
                   'timeout_seconds', 'measurement_timeout_seconds', 'setup_timeout_seconds',
                   'warmup_timeout_seconds', 'profiling_timeout_seconds',
                   'nsys_timeout_seconds', 'ncu_timeout_seconds', 'proton_timeout_seconds', 'timeout_multiplier',
                   'execution_mode', 'launch_via', 'nproc_per_node', 'nnodes', 'rdzv_backend',
                   'rdzv_endpoint', 'env_passthrough', 'target_extra_args', 'multi_gpu_required',
                   'target_label']:
            value = getattr(config, key, None)
            # For percentiles, CRITICAL: always include it and ensure it's a list (never None)
            if key == 'percentiles':
                # Ensure percentiles is always a list before serialization
                if value is None or not isinstance(value, list):
                    value = [25, 50, 75, 99]
                config_dict[key] = value
            elif key == 'execution_mode':
                if value is not None:
                    config_dict[key] = value.value if isinstance(value, ExecutionMode) else value
            elif key == 'launch_via':
                if value is not None:
                    config_dict[key] = value.value if isinstance(value, LaunchVia) else value
            elif value is not None:
                config_dict[key] = value
        
        # CRITICAL: Ensure percentiles is always in config_dict
        if 'percentiles' not in config_dict:
            config_dict['percentiles'] = [25, 50, 75, 99]
        
        def _is_simple(value: Any) -> bool:
            """Return True for JSON-serializable scalar/collection values."""
            if isinstance(value, (str, int, float, bool)) or value is None:
                return True
            if isinstance(value, (list, tuple)):
                return all(_is_simple(v) for v in value)
            if isinstance(value, dict):
                return all(
                    isinstance(k, (str, int, float, bool)) and _is_simple(v)
                    for k, v in value.items()
                )
            return False

        initial_state = {
            key: value
            for key, value in getattr(benchmark, "__dict__", {}).items()
            if not str(key).startswith("_") and _is_simple(value)
        }

        # Prepare input JSON
        input_data = {
            "benchmark_module_path": str(module_path),
            "benchmark_class_name": benchmark_class,
            "config_dict": config_dict,
            "device": str(self.device) if self.device else None,
            "initial_state": initial_state or None,
        }
        
        # Spawn subprocess using isolated runner
        runner_script = Path(__file__).parent / "isolated_runner.py"
        if not runner_script.exists():
            errors.append("isolated_runner.py not found - falling back to threading")
            return self._benchmark_with_threading(benchmark, config)
        
        # Use measurement_timeout_seconds (or fallback to timeout_seconds for backward compatibility)
        measurement_timeout = getattr(config, 'measurement_timeout_seconds', config.timeout_seconds)
        start_time = time.time()
        
        try:
            import signal
            process = subprocess.Popen(
                [sys.executable, str(runner_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create new process group for reliable killing
            )
            
            # Send input JSON and wait for result
            input_json = json.dumps(input_data)
            stdout, stderr = process.communicate(input=input_json, timeout=measurement_timeout)
            
            if process.returncode != 0:
                error_msg = f"Subprocess exited with code {process.returncode}"
                errors.append(error_msg)
                if stderr:
                    stderr_preview = stderr[:500] if len(stderr) > 500 else stderr
                    errors.append(f"Stderr: {stderr_preview}")
                    if LOGGER_AVAILABLE:
                        logger.error(f"Benchmark '{benchmark_name}' subprocess failed with return code {process.returncode}")
                        logger.error(f"Stderr preview: {stderr_preview}")
                times_ms = cast(List[float], [])
            else:
                # Parse JSON result from isolated runner
                try:
                    result_dict = json.loads(stdout)
                    if result_dict.get("success") and result_dict.get("result_json"):
                        # Deserialize Pydantic BenchmarkResult from JSON
                        result_json_str = result_dict["result_json"]
                        benchmark_result = PydanticBenchmarkResult.model_validate_json(result_json_str)
                        
                        # Extract timing data
                        if benchmark_result.timing.raw_times_ms:
                            times_ms = benchmark_result.timing.raw_times_ms
                        else:
                            # Reconstruct from statistics if raw times not available
                            mean_time = benchmark_result.timing.mean_ms
                            std_time = benchmark_result.timing.std_ms
                            iterations = benchmark_result.timing.iterations
                            import numpy as np
                            synthetic_times = np.random.normal(mean_time, std_time, iterations)
                            synthetic_times = np.clip(
                                synthetic_times,
                                benchmark_result.timing.min_ms,
                                benchmark_result.timing.max_ms
                            )
                            times_ms = synthetic_times.tolist()
                        
                        # Extract memory data
                        if benchmark_result.memory:
                            memory_peak_mb = benchmark_result.memory.peak_mb
                            memory_allocated_mb = benchmark_result.memory.allocated_mb
                        
                        # Extract errors
                        errors.extend(benchmark_result.errors)
                        
                        # Extract profiling artifacts
                        if benchmark_result.artifacts:
                            if benchmark_result.artifacts.nsys_rep:
                                profiling_outputs['nsys'] = benchmark_result.artifacts.nsys_rep
                            if benchmark_result.artifacts.ncu_rep:
                                profiling_outputs['ncu'] = benchmark_result.artifacts.ncu_rep
                            if benchmark_result.artifacts.torch_trace_json:
                                profiling_outputs['torch'] = benchmark_result.artifacts.torch_trace_json
                        
                        # Extract profiler metrics
                        if benchmark_result.profiler_metrics:
                            if benchmark_result.profiler_metrics.nsys:
                                nsys_metrics = benchmark_result.profiler_metrics.nsys.to_dict()
                            if benchmark_result.profiler_metrics.ncu:
                                ncu_metrics = benchmark_result.profiler_metrics.ncu.to_dict()
                            if benchmark_result.profiler_metrics.proton:
                                proton_metrics = benchmark_result.profiler_metrics.proton.to_dict()
                    else:
                        errors.extend(result_dict.get("errors", ["Subprocess execution failed"]))
                        times_ms = cast(List[float], [])
                except json.JSONDecodeError as e:
                    errors.append(f"Failed to parse subprocess output: {e}")
                    errors.append(f"Output: {stdout[:500]}")
                    times_ms = cast(List[float], [])
                except Exception as e:
                    errors.append(f"Error processing subprocess result: {e}")
                    errors.append(f"Traceback: {traceback.format_exc()}")
                    times_ms = cast(List[float], [])
        
        except subprocess.TimeoutExpired:
            # TIMEOUT - kill the process group
            elapsed_time = time.time() - start_time
            
            logger.error("=" * 80)
            logger.error("TIMEOUT: Benchmark execution exceeded timeout limit")
            logger.error("=" * 80)
            logger.error(f"   Benchmark: {benchmark_name}")
            logger.error(f"   Stage: measurement (benchmark iterations)")
            logger.error(f"   Timeout limit: {measurement_timeout} seconds")
            logger.error(f"   Elapsed time: {elapsed_time:.2f} seconds")
            logger.error(f"   Config: iterations={config.iterations}, warmup={config.warmup}")
            logger.error(f"   Status: Benchmark subprocess did not complete within timeout period")
            logger.error("")
            logger.error("   Possible causes:")
            logger.error("   - Benchmark is too slow for current timeout")
            logger.error("   - Benchmark is hanging or deadlocked")
            logger.error("   - GPU is under heavy load from other processes")
            logger.error("")
            logger.error("   Suggested actions:")
            logger.error(f"   - Increase timeout: config.measurement_timeout_seconds = {measurement_timeout * 2}")
            logger.error(f"   - Use timeout multiplier: config.timeout_multiplier = 2.0")
            logger.error("   - Check for GPU resource contention")
            logger.error("   - Review benchmark code for potential deadlocks")
            logger.error("")
            logger.error("   Action: Terminating subprocess to free GPU resources")
            logger.error("=" * 80)
            
            try:
                # Kill the process group (only the child, not parent)
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                # Wait a bit for graceful termination
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            except (ProcessLookupError, OSError):
                # Process already terminated
                pass
            
            timeout_error_msg = (
                f"TIMEOUT: Benchmark measurement stage exceeded timeout of {measurement_timeout} seconds "
                f"(ran for {elapsed_time:.2f}s). "
                f"Consider increasing measurement_timeout_seconds or using timeout_multiplier."
            )
            errors.append(timeout_error_msg)
            times_ms = cast(List[float], [])
            
            # Create a result with timeout information even though we failed
            if not times_ms:
                return self._create_timeout_result(
                    stage="measurement",
                    duration=elapsed_time,
                    limit=measurement_timeout,
                    errors=errors,
                    benchmark_name=benchmark_name,
                    config=config,
                )
        
        except Exception as e:
            error_msg = f"Subprocess execution failed: {str(e)}"
            errors.append(error_msg)
            if LOGGER_AVAILABLE:
                logger.error(f"Benchmark '{benchmark_name}' failed during subprocess execution")
                logger.error(f"Error: {error_msg}")
            times_ms = cast(List[float], [])
        
        # Don't raise if we already returned a timeout result
        if not times_ms and 'timeout_result' not in locals():
            # Build comprehensive error message with context
            config_summary = (
                f"iterations={config.iterations}, warmup={config.warmup}, "
                f"timeout={getattr(config, 'measurement_timeout_seconds', config.timeout_seconds)}s"
            )
            error_details = [
                f"Benchmark: {benchmark_name}",
                f"Config: {config_summary}",
                f"Errors: {', '.join(errors)}",
            ]
            # Include partial results if we have any
            if memory_peak_mb is not None:
                error_details.append(f"Partial memory stats: peak={memory_peak_mb:.2f}MB")
            if profiling_outputs:
                error_details.append(f"Profiling artifacts collected: {list(profiling_outputs.keys())}")

            skip_reason = _extract_skip_reason_from_messages(errors)
            if skip_reason:
                skip_message = skip_reason if skip_reason.upper().startswith("SKIPPED") else f"SKIPPED: {skip_reason}"
                if LOGGER_AVAILABLE:
                    logger.warning("Benchmark '%s' skipped: %s", benchmark_name, skip_message)
                raise RuntimeError(skip_message)

            error_message = "\n".join(error_details)
            if LOGGER_AVAILABLE:
                logger.error("=" * 80)
                logger.error("BENCHMARK FAILED")
                logger.error("=" * 80)
                for detail in error_details:
                    logger.error(f"  {detail}")
                logger.error("=" * 80)
            
            raise RuntimeError(error_message)
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)
        custom_metrics = self._resolve_custom_metrics(benchmark)
        if custom_metrics:
            result.custom_metrics = custom_metrics
        custom_metrics = self._resolve_custom_metrics(benchmark)
        if custom_metrics:
            result.custom_metrics = custom_metrics
        custom_metrics = self._resolve_custom_metrics(benchmark)
        if custom_metrics:
            result.custom_metrics = custom_metrics
        self._attach_throughput_metrics(result, benchmark)
        self._attach_throughput_metrics(result, benchmark)
        
        # Add inference timing if available
        if inference_timing_data and inference_timing_data.get("ttft_times_ms") and inference_timing_data.get("tpot_times_ms"):
            inference_timing = self._capture_inference_timing(
                inference_timing_data["ttft_times_ms"],
                inference_timing_data["tpot_times_ms"],
                config
            )
            if inference_timing:
                result.inference_timing = inference_timing
        
        # Add memory stats
        if memory_peak_mb is not None or memory_allocated_mb is not None:
            result.memory = MemoryStats(
                peak_mb=memory_peak_mb,
                allocated_mb=memory_allocated_mb,
                reserved_mb=None,
                schemaVersion="1.0",
            )
        
        # Add errors
        result.errors = errors
        
        # Add profiling artifacts
        if profiling_outputs:
            result.artifacts = ProfilerArtifacts(
                nsys_rep=profiling_outputs.get('nsys_rep') or profiling_outputs.get('nsys'),
                ncu_rep=profiling_outputs.get('ncu_rep') or profiling_outputs.get('ncu'),
                torch_trace_json=profiling_outputs.get('torch') or profiling_outputs.get('pytorch_trace'),
                proton_report=profiling_outputs.get('proton_report') or profiling_outputs.get('proton'),
                schemaVersion="1.0",
            )
        
        # Add profiler metrics
        nsys_metrics_obj = None
        ncu_metrics_obj = None
        proton_metrics_obj = None
        
        if nsys_metrics:
            nsys_metrics_obj = NsysMetrics(
                total_gpu_time_ms=nsys_metrics.get('nsys_total_gpu_time_ms'),
                raw_metrics={k.replace('nsys_', ''): v for k, v in nsys_metrics.items() 
                            if not k.startswith('nsys_total_gpu_time_ms')},
                schemaVersion="1.0",
            )
        
        if ncu_metrics:
            ncu_metrics_obj = NcuMetrics(
                kernel_time_ms=ncu_metrics.get('ncu_kernel_time_ms'),
                sm_throughput_pct=ncu_metrics.get('ncu_sm_throughput_pct'),
                dram_throughput_pct=ncu_metrics.get('ncu_dram_throughput_pct'),
                l2_throughput_pct=ncu_metrics.get('ncu_l2_throughput_pct'),
                occupancy_pct=ncu_metrics.get('ncu_occupancy_pct'),
                raw_metrics={k.replace('ncu_', ''): v for k, v in ncu_metrics.items() 
                            if not k.startswith(('ncu_kernel_time_ms', 'ncu_sm_throughput_pct', 
                                                'ncu_dram_throughput_pct', 'ncu_l2_throughput_pct', 'ncu_occupancy_pct'))},
                schemaVersion="1.0",
            )
        
        if proton_metrics:
            occ_field = proton_metrics.get('proton_occupancy_limited_kernels_list', proton_metrics.get('proton_occupancy_limited_kernels', ""))
            if isinstance(occ_field, list):
                occupancy = [str(item) for item in occ_field if item]
            else:
                occupancy = [item for item in occ_field.split(",") if item] if isinstance(occ_field, str) else []
            summary_stats: Dict[str, Any] = {}
            summary_from_report = proton_metrics.get('proton_summary_stats')
            if isinstance(summary_from_report, dict):
                summary_stats.update(summary_from_report)
            summary_stats.update({
                k.replace('proton_', ''): v for k, v in proton_metrics.items()
                if k.startswith('proton_') and k not in {'proton_kernel_count', 'proton_occupancy_limited_kernels', 'proton_top_kernels', 'proton_summary_stats', 'proton_kernel_summaries', 'proton_occupancy_limited_kernels_list'}
            })
            kernel_summaries = proton_metrics.get('proton_kernel_summaries')
            if not isinstance(kernel_summaries, list):
                kernel_summaries = []
            proton_metrics_obj = ProtonMetrics(
                kernel_count=proton_metrics.get('proton_kernel_count'),
                occupancy_limited_kernels=occupancy,
                summary_stats=summary_stats,
                kernel_summaries=kernel_summaries,
                schemaVersion="1.0",
            )
        
        if nsys_metrics_obj or ncu_metrics_obj or proton_metrics_obj:
            result.profiler_metrics = ProfilerMetrics(
                nsys=nsys_metrics_obj,
                ncu=ncu_metrics_obj,
                proton=proton_metrics_obj,
                torch=None,
                schemaVersion="1.0",
            )
        
        self._annotate_launch_metadata(result, config)
        return result
    
    def _benchmark_with_threading(self, benchmark: BaseBenchmark, config: BenchmarkConfig) -> PydanticBenchmarkResult:
        """Run benchmark using threading (alternative to subprocess method)."""
        import inspect
        
        errors = []
        memory_peak_mb = None
        memory_allocated_mb = None
        profiling_outputs = {}
        nsys_metrics = {}
        ncu_metrics = {}
        proton_metrics = {}
        times_ms = cast(List[float], [])
        inference_timing_data = None
        
        # Get benchmark name for error messages (same as subprocess path)
        benchmark_class = benchmark.__class__.__name__
        benchmark_name = getattr(benchmark, '__name__', None) or getattr(benchmark, 'name', None) or benchmark_class
        
        # Use a lock to prevent teardown from running while benchmark is executing
        execution_lock = threading.Lock()
        execution_complete = threading.Event()
        teardown_called = threading.Event()  # Track if teardown has been called
        
        # Store timeout result if one occurs (needs to be accessible from outer scope)
        timeout_result_storage: List[Optional[PydanticBenchmarkResult]] = [None]
        stage_watchdog: Dict[str, Dict[str, Any]] = {
            'setup': {'status': 'pending', 'duration': 0.0},
            'warmup': {'status': 'pending', 'duration': 0.0},
            'measurement': {'status': 'pending', 'duration': 0.0},
            'profiling': {'status': 'pending', 'duration': 0.0},
        }
        stage_start_times: Dict[str, float] = {}

        def start_stage(stage: str) -> None:
            stage_start_times[stage] = time.time()
            stage_watchdog[stage] = {'status': 'running'}

        def finish_stage(stage: str, status: str = "completed") -> None:
            start = stage_start_times.get(stage)
            duration = time.time() - start if start else 0.0
            stage_watchdog[stage] = {'status': status, 'duration': duration}

        def mark_stage(stage: str, status: str) -> None:
            entry = stage_watchdog.get(stage, {})
            entry['status'] = status
            stage_watchdog[stage] = entry
        
        def run_benchmark_internal():
            """Internal benchmark execution function."""
            nonlocal times_ms, memory_peak_mb, memory_allocated_mb, profiling_outputs, errors, nsys_metrics, ncu_metrics, timeout_result_storage, inference_timing_data
            
            with execution_lock:  # Acquire lock during execution
                try:
                    # Ensure the CUDA primary context exists before any kernels run.
                    if torch.cuda.is_available():
                        torch.cuda.init()
                        try:
                            target_device = None
                            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                                target_device = self.device
                            if target_device is None:
                                target_device = torch.device(torch.cuda.current_device())
                            torch.cuda.set_device(target_device)
                            # Tiny ops to bind the primary context and initialize cuBLAS handles
                            torch.empty(1, device=target_device).add_(1)
                            torch.ones((1, 1), device=target_device).matmul(
                                torch.ones((1, 1), device=target_device)
                            )
                            torch.cuda.synchronize(target_device)
                        except Exception:
                            # Best-effort context warm-up; continue even if it fails
                            pass

                    # Apply per-target CLI overrides (e.g., backend selection) before setup.
                    self._apply_target_overrides(benchmark, config)

                    # Setup - this may include CUDA extension compilation OR torch.compile()
                    # IMPORTANT: Setup can hang, so we need actual timeout enforcement
                    import time
                    setup_start_time = time.time()
                    setup_timeout = config.get_effective_timeout('setup')
                    
                    if setup_timeout is not None:
                        # Setup has explicit timeout - enforce it with threading timeout
                        setup_complete = threading.Event()
                        setup_error: List[Optional[Exception]] = [None]
                        
                        def run_setup():
                            try:
                                start_stage('setup')
                                benchmark.setup()
                                finish_stage('setup')
                                setup_complete.set()
                            except Exception as e:
                                setup_error[0] = e
                                finish_stage('setup', status='error')
                                setup_complete.set()
                        
                        setup_thread = threading.Thread(target=run_setup, daemon=True)
                        setup_thread.start()
                        setup_thread.join(timeout=setup_timeout)
                        
                        if not setup_complete.is_set():
                            # Setup timed out
                            setup_time = time.time() - setup_start_time
                            finish_stage('setup', status='timeout')
                            timeout_error = TimeoutError(
                                f"Setup exceeded timeout of {setup_timeout}s (ran for {setup_time:.1f}s)"
                            )
                            errors.append(str(timeout_error))
                            # Create timeout result
                            timeout_result_storage[0] = self._create_timeout_result(
                                stage="setup",
                                duration=setup_time,
                                limit=setup_timeout,
                                errors=errors,
                                benchmark_name=benchmark_name,
                                config=config,
                                watchdog=stage_watchdog,
                            )
                            return  # Exit early, timeout_result will be checked outside
                        
                        if setup_error[0]:
                            raise setup_error[0]
                        
                        setup_time = time.time() - setup_start_time
                        if setup_time > setup_timeout * 0.8:  # Warn if setup takes >80% of timeout
                            logger.warning(f"Setup took {setup_time:.1f}s (near timeout limit of {setup_timeout}s)")
                    else:
                        # No explicit setup timeout - just run it
                        start_stage('setup')
                        benchmark.setup()
                        finish_stage('setup')
                        setup_time = time.time() - setup_start_time
                        # Warn if setup is suspiciously long (even without timeout)
                        measurement_timeout = config.get_effective_timeout('measurement')
                        if measurement_timeout and setup_time > measurement_timeout * 0.5:
                            logger.warning(f"Setup took {setup_time:.1f}s (consider setting setup_timeout_seconds)")
                    
                    # Warmup with timeout enforcement
                    warmup_timeout = config.get_effective_timeout('warmup')
                    if warmup_timeout is not None and config.warmup > 0:
                        warmup_start_time = time.time()
                        warmup_complete = threading.Event()
                        warmup_error: List[Optional[Exception]] = [None]
                        
                        def run_warmup():
                            try:
                                start_stage('warmup')
                                self._warmup(benchmark.benchmark_fn, config.warmup)
                                finish_stage('warmup')
                                warmup_complete.set()
                            except Exception as e:
                                warmup_error[0] = e
                                finish_stage('warmup', status='error')
                                warmup_complete.set()
                        
                        warmup_thread = threading.Thread(target=run_warmup, daemon=True)
                        warmup_thread.start()
                        warmup_thread.join(timeout=warmup_timeout)
                        
                        if not warmup_complete.is_set():
                            # Warmup timed out
                            warmup_time = time.time() - warmup_start_time
                            finish_stage('warmup', status='timeout')
                            timeout_error = TimeoutError(
                                f"Warmup exceeded timeout of {warmup_timeout}s (ran for {warmup_time:.1f}s)"
                            )
                            errors.append(str(timeout_error))
                            # Create timeout result
                            timeout_result_storage[0] = self._create_timeout_result(
                                stage="warmup",
                                duration=warmup_time,
                                limit=warmup_timeout,
                                errors=errors,
                                benchmark_name=benchmark_name,
                                config=config,
                                watchdog=stage_watchdog,
                            )
                            return  # Exit early, timeout_result will be checked outside
                        
                        if warmup_error[0]:
                            raise warmup_error[0]
                    else:
                        # No warmup timeout - just run it
                        if config.warmup > 0:
                            start_stage('warmup')
                            self._warmup(benchmark.benchmark_fn, config.warmup)
                            finish_stage('warmup')
                        else:
                            mark_stage('warmup', 'skipped')
                    
                    # Memory tracking: Use context manager to track peak memory during benchmark execution
                    start_stage('measurement')
                    with self._memory_tracking(config) as mem_result:
                        # Benchmark using selected mode
                        # Note: nsys/ncu profiling wraps the entire process, so it's handled separately
                        if config.enable_profiling and (config.enable_nsys or config.enable_ncu or config.enable_proton):
                            if stage_watchdog['profiling']['status'] == 'pending':
                                start_stage('profiling')
                            # Delegate to unified profiling orchestration with timeout enforcement
                            profiling_timeout = config.get_effective_timeout('profiling')
                            if PROFILING_RUNNER_AVAILABLE:
                                try:
                                    from core.profiling.profiling_runner import run_profiling_orchestration
                                    import time
                                    profiling_start_time = time.time()
                                    
                                    # Run profiling with timeout enforcement
                                    if profiling_timeout is not None:
                                        profiling_complete = threading.Event()
                                        profiling_result: List[Optional[Dict[str, Any]]] = [None]
                                        profiling_error: List[Optional[Exception]] = [None]
                                        
                                        def run_profiling():
                                            try:
                                                # Wrap _benchmark_without_profiling to match expected signature
                                                def timing_wrapper(fn: Callable, cfg: BenchmarkConfig) -> List[float]:
                                                    times, _ = self._benchmark_without_profiling(fn, cfg)
                                                    return times
                                                
                                                result = run_profiling_orchestration(
                                                    benchmark, config,
                                                    timing_fn=timing_wrapper,
                                                    output_dir=Path(config.profiling_output_dir) if config.profiling_output_dir else None
                                                )
                                                profiling_result[0] = result
                                                profiling_complete.set()
                                            except Exception as e:
                                                profiling_error[0] = e
                                                profiling_complete.set()
                                        
                                        profiling_thread = threading.Thread(target=run_profiling, daemon=True)
                                        profiling_thread.start()
                                        profiling_thread.join(timeout=profiling_timeout)
                                        
                                        if not profiling_complete.is_set():
                                            # Profiling timed out
                                            profiling_time = time.time() - profiling_start_time
                                            timeout_error = TimeoutError(
                                                f"Profiling exceeded timeout of {profiling_timeout}s (ran for {profiling_time:.1f}s)"
                                            )
                                            errors.append(str(timeout_error))
                                            finish_stage('profiling', status='timeout')
                                            # Create timeout result
                                            timeout_result_storage[0] = self._create_timeout_result(
                                                stage="profiling",
                                                duration=profiling_time,
                                                limit=profiling_timeout,
                                                errors=errors,
                                                benchmark_name=benchmark_name,
                                                config=config,
                                                watchdog=stage_watchdog,
                                            )
                                            return  # Exit early, timeout_result will be checked outside
                                        
                                        if profiling_error[0]:
                                            finish_stage('profiling', status='error')
                                            raise profiling_error[0]
                                        
                                        prof_result = profiling_result[0]
                                        if stage_watchdog['profiling']['status'] == 'running':
                                            finish_stage('profiling')
                                    else:
                                        # No profiling timeout - just run it
                                        # Wrap _benchmark_without_profiling to match expected signature
                                        def timing_wrapper_no_timeout(fn: Callable, cfg: BenchmarkConfig) -> List[float]:
                                            times, _ = self._benchmark_without_profiling(fn, cfg)
                                            return times
                                        
                                        prof_result = run_profiling_orchestration(
                                            benchmark, config,
                                            timing_fn=timing_wrapper_no_timeout,
                                            output_dir=Path(config.profiling_output_dir) if config.profiling_output_dir else None
                                        )
                                        finish_stage('profiling')
                                    
                                    if prof_result:
                                        times_ms = prof_result.get("times_ms", [])
                                        if "profiling_outputs" in prof_result:
                                            profiling_outputs.update(prof_result.get("profiling_outputs", {}))
                                        nsys_metrics = prof_result.get("nsys_metrics", {})
                                        ncu_metrics = prof_result.get("ncu_metrics", {})
                                        proton_metrics = prof_result.get("proton_metrics", {})
                                except ImportError:
                                    # Fallback if orchestration function not available
                                    if stage_watchdog['profiling']['status'] == 'pending':
                                        start_stage('profiling')
                                    times_ms, profiling_outputs = self._benchmark_with_profiling(
                                        benchmark.benchmark_fn, config
                                    )
                                    finish_stage('profiling')
                                    nsys_metrics = {}
                                    ncu_metrics = {}
                                    proton_metrics = {}
                            else:
                                # Fallback to PyTorch profiler if profiling runner not available
                                if stage_watchdog['profiling']['status'] == 'pending':
                                    start_stage('profiling')
                                times_ms, profiling_outputs = self._benchmark_with_profiling(
                                    benchmark.benchmark_fn, config
                                )
                                finish_stage('profiling')
                                nsys_metrics = {}
                                ncu_metrics = {}
                                proton_metrics = {}
                        elif config.enable_profiling:
                            start_stage('profiling')
                            times_ms, profiling_outputs = self._benchmark_with_profiling(
                                benchmark.benchmark_fn, config
                            )
                            finish_stage('profiling')
                        else:
                            mark_stage('profiling', 'skipped')
                            times_ms, inference_timing_data = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
                    
                    # Extract memory tracking results from context manager
                    if mem_result is not None:
                        if not isinstance(mem_result, MemoryStats):
                            raise TypeError(f"Expected MemoryStats, got {type(mem_result)}")
                        memory_peak_mb = mem_result.peak_mb
                        memory_allocated_mb = mem_result.allocated_mb
                    
                    # Validate result
                    validation_error = benchmark.validate_result()
                    if validation_error:
                        errors.append(f"Validation failed: {validation_error}")
                    
                    if stage_watchdog['measurement']['status'] == 'running':
                        finish_stage('measurement')
                
                except Exception as e:
                    error_msg = str(e) or repr(e)
                    # Handle generator errors gracefully (common with torch.compile)
                    if "generator didn't stop after throw" in error_msg:
                        errors.append(f"Benchmark execution failed: Generator error (likely from torch.compile or async operations)")
                    else:
                        errors.append(f"Benchmark execution failed: {error_msg}")
                        errors.append(f"Traceback: {traceback.format_exc()}")
                    times_ms = cast(List[float], [])
                    if stage_watchdog['measurement']['status'] == 'running':
                        finish_stage('measurement', status='error')
                finally:
                    # Mark execution as complete before teardown
                    execution_complete.set()
                    
                    # Teardown is now safe to call - we hold the lock
                    # Only call teardown once - set flag to prevent double invocation
                    if not teardown_called.is_set():
                        try:
                            benchmark.teardown()
                            teardown_called.set()
                        except Exception as e:
                            errors.append(f"Teardown failed: {str(e)}")
                            teardown_called.set()  # Mark as called even on error
                    
                    # Force cleanup (only if enabled to avoid distorting timings)
                    if config.enable_cleanup:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
        
        # ALWAYS run with timeout (required, default 15 seconds). Thread-mode benchmarks execute
        # inside a managed executor so we can recycle workers if a benchmark hangs.
        measurement_timeout = getattr(config, 'measurement_timeout_seconds', config.timeout_seconds)
        thread_start_time = time.time()
        self._ensure_thread_executor()
        future = self._thread_executor.submit(run_benchmark_internal)
        try:
            future.result(timeout=measurement_timeout)
            elapsed_time = time.time() - thread_start_time
        except _FuturesTimeoutError:
            elapsed_time = time.time() - thread_start_time
            future.cancel()
            finish_stage('measurement', status='timeout')
            
            if timeout_result_storage[0] is not None:
                self._reset_thread_executor()
                return timeout_result_storage[0]
            
            logger.error("=" * 80)
            logger.error("TIMEOUT: Benchmark execution exceeded timeout limit")
            logger.error("=" * 80)
            logger.error(f"   Benchmark: {benchmark_name}")
            logger.error(f"   Stage: measurement (benchmark iterations)")
            logger.error(f"   Timeout limit: {measurement_timeout} seconds")
            logger.error(f"   Elapsed time: {elapsed_time:.2f} seconds")
            logger.error(f"   Config: iterations={config.iterations}, warmup={config.warmup}")
            logger.error(f"   Status: Benchmark did not complete within timeout period")
            logger.error("")
            logger.error("   Possible causes:")
            logger.error("   - Benchmark is too slow for current timeout")
            logger.error("   - Benchmark is hanging or deadlocked")
            logger.error("   - CUDA kernel is hung (cannot be interrupted from Python)")
            logger.error("   - GPU is under heavy load from other processes")
            logger.error("")
            logger.error("   Suggested actions:")
            logger.error(f"   - Increase timeout: config.measurement_timeout_seconds = {measurement_timeout * 2}")
            logger.error(f"   - Use timeout multiplier: config.timeout_multiplier = 2.0")
            logger.error("   - Check for GPU resource contention")
            logger.error("   - Review benchmark code for potential deadlocks")
            logger.error("   - Use subprocess mode (more reliable): config.use_subprocess = True")
            logger.error("")
            logger.error("   WARNING: Threading timeout cannot force-stop hung CUDA kernels.")
            logger.error("=" * 80)
            
            timeout_error_msg = (
                f"TIMEOUT: Benchmark measurement stage exceeded timeout of {measurement_timeout} seconds "
                f"(ran for {elapsed_time:.2f}s). "
                f"Consider increasing measurement_timeout_seconds or using timeout_multiplier. "
                f"Thread-mode cannot force-stop hung CUDA kernels - consider subprocess mode for stricter isolation."
            )
            errors.append(timeout_error_msg)
            times_ms = cast(List[float], [])
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
            gc.collect()
            
            timeout_result = self._create_timeout_result(
                stage="measurement",
                duration=elapsed_time,
                limit=measurement_timeout,
                errors=errors,
                benchmark_name=benchmark_name,
                config=config,
                watchdog=stage_watchdog,
            )
            self._reset_thread_executor()
            return timeout_result
        finally:
            self._reset_thread_executor()
        
        # Check if a timeout occurred in setup/warmup/profiling stages
        if timeout_result_storage[0] is not None:
            return timeout_result_storage[0]
        
        if execution_complete.is_set():
            # Thread completed normally - results are already set
            pass
        else:
            # Thread completed but execution_complete not set - something went wrong
            if not times_ms:
                errors.append("Benchmark execution completed but no results collected")
        # Don't print success message for normal completion - only print on timeout/failure
        
        if not times_ms:
            # Build comprehensive error message with context
            config_summary = (
                f"iterations={config.iterations}, warmup={config.warmup}, "
                f"timeout={getattr(config, 'measurement_timeout_seconds', config.timeout_seconds)}s"
            )
            error_details = [
                f"Benchmark: {benchmark_name}",
                f"Config: {config_summary}",
                f"Errors: {', '.join(errors)}",
            ]
            # Include partial results if we have any
            if memory_peak_mb is not None:
                error_details.append(f"Partial memory stats: peak={memory_peak_mb:.2f}MB")
            if profiling_outputs:
                error_details.append(f"Profiling artifacts collected: {list(profiling_outputs.keys())}")

            skip_reason = _extract_skip_reason_from_messages(errors)
            if skip_reason:
                skip_message = skip_reason if skip_reason.upper().startswith("SKIPPED") else f"SKIPPED: {skip_reason}"
                if LOGGER_AVAILABLE:
                    logger.warning("Benchmark '%s' skipped: %s", benchmark_name, skip_message)
                raise RuntimeError(skip_message)

            error_message = "\n".join(error_details)
            if LOGGER_AVAILABLE:
                logger.error("=" * 80)
                logger.error("BENCHMARK FAILED")
                logger.error("=" * 80)
                for detail in error_details:
                    logger.error(f"  {detail}")
                logger.error("=" * 80)
            
            raise RuntimeError(error_message)
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)

        # Attach benchmark-specific metrics and throughput (parity with subprocess path)
        custom_metrics = self._resolve_custom_metrics(benchmark)
        if custom_metrics:
            result.custom_metrics = custom_metrics
        self._attach_throughput_metrics(result, benchmark)

        # Add inference timing if available
        if inference_timing_data and inference_timing_data.get("ttft_times_ms") and inference_timing_data.get("tpot_times_ms"):
            inference_timing = self._capture_inference_timing(
                inference_timing_data["ttft_times_ms"],
                inference_timing_data["tpot_times_ms"],
                config
            )
            if inference_timing:
                result.inference_timing = inference_timing
        
        # Add memory stats
        if memory_peak_mb is not None or memory_allocated_mb is not None:
            result.memory = MemoryStats(
                peak_mb=memory_peak_mb,
                allocated_mb=memory_allocated_mb,
                reserved_mb=None,
                schemaVersion="1.0",
            )
        
        # Add errors
        result.errors = errors
        
        # Add profiling artifacts
        if profiling_outputs:
            result.artifacts = ProfilerArtifacts(
                nsys_rep=profiling_outputs.get('nsys_rep') or profiling_outputs.get('nsys'),
                ncu_rep=profiling_outputs.get('ncu_rep') or profiling_outputs.get('ncu'),
                torch_trace_json=profiling_outputs.get('torch') or profiling_outputs.get('pytorch_trace'),
                proton_report=profiling_outputs.get('proton_report') or profiling_outputs.get('proton'),
                schemaVersion="1.0",
            )
        
        # Add profiler metrics
        nsys_metrics_obj = None
        ncu_metrics_obj = None
        proton_metrics_obj = None
        
        if nsys_metrics:
            nsys_metrics_obj = NsysMetrics(
                total_gpu_time_ms=nsys_metrics.get('nsys_total_gpu_time_ms'),
                raw_metrics={k.replace('nsys_', ''): v for k, v in nsys_metrics.items() 
                            if not k.startswith('nsys_total_gpu_time_ms')},
                schemaVersion="1.0",
            )
        
        if ncu_metrics:
            ncu_metrics_obj = NcuMetrics(
                kernel_time_ms=ncu_metrics.get('ncu_kernel_time_ms'),
                sm_throughput_pct=ncu_metrics.get('ncu_sm_throughput_pct'),
                dram_throughput_pct=ncu_metrics.get('ncu_dram_throughput_pct'),
                l2_throughput_pct=ncu_metrics.get('ncu_l2_throughput_pct'),
                occupancy_pct=ncu_metrics.get('ncu_occupancy_pct'),
                raw_metrics={k.replace('ncu_', ''): v for k, v in ncu_metrics.items() 
                            if not k.startswith(('ncu_kernel_time_ms', 'ncu_sm_throughput_pct', 
                                                'ncu_dram_throughput_pct', 'ncu_l2_throughput_pct', 'ncu_occupancy_pct'))},
                schemaVersion="1.0",
            )
        
        if proton_metrics:
            occ_field = proton_metrics.get('proton_occupancy_limited_kernels_list', proton_metrics.get('proton_occupancy_limited_kernels', ""))
            if isinstance(occ_field, list):
                occupancy = [str(item) for item in occ_field if item]
            else:
                occupancy = [item for item in occ_field.split(",") if item] if isinstance(occ_field, str) else []
            summary_stats: Dict[str, Any] = {}
            summary_from_report = proton_metrics.get('proton_summary_stats')
            if isinstance(summary_from_report, dict):
                summary_stats.update(summary_from_report)
            summary_stats.update({
                k.replace('proton_', ''): v for k, v in proton_metrics.items()
                if k.startswith('proton_') and k not in {'proton_kernel_count', 'proton_occupancy_limited_kernels', 'proton_top_kernels', 'proton_summary_stats', 'proton_kernel_summaries', 'proton_occupancy_limited_kernels_list'}
            })
            kernel_summaries = proton_metrics.get('proton_kernel_summaries')
            if not isinstance(kernel_summaries, list):
                kernel_summaries = []
            proton_metrics_obj = ProtonMetrics(
                kernel_count=proton_metrics.get('proton_kernel_count'),
                occupancy_limited_kernels=occupancy,
                summary_stats=summary_stats,
                kernel_summaries=kernel_summaries,
                schemaVersion="1.0",
            )
        
        if nsys_metrics_obj or ncu_metrics_obj or proton_metrics_obj:
            result.profiler_metrics = ProfilerMetrics(
                nsys=nsys_metrics_obj,
                ncu=ncu_metrics_obj,
                proton=proton_metrics_obj,
                torch=None,
                schemaVersion="1.0",
            )
        
        self._annotate_launch_metadata(result, config)
        return result
    
    def _benchmark_with_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> tuple[List[float], Dict[str, str]]:
        """Benchmark with profiling enabled."""
        profiling_outputs = {}
        
        # Create profiling output directory
        if config.profiling_output_dir:
            prof_dir = Path(config.profiling_output_dir)
            prof_dir.mkdir(parents=True, exist_ok=True)
        else:
            prof_dir = Path("profiling_results")
            prof_dir.mkdir(parents=True, exist_ok=True)
        
        # Try PyTorch profiler first (best for Python benchmarks)
        try:
            import torch.profiler
            
            torch_profiler_kwargs = {}
            try:
                from core.profiling import profiler_config as profiler_config_mod
                profiler_cfg = None
                if hasattr(profiler_config_mod, "build_profiler_config_from_benchmark"):
                    profiler_cfg = profiler_config_mod.build_profiler_config_from_benchmark(config)
                elif hasattr(profiler_config_mod, "DEFAULT_PROFILER_CONFIG"):
                    profiler_cfg = profiler_config_mod.DEFAULT_PROFILER_CONFIG
                if profiler_cfg is not None:
                    torch_profiler_kwargs = profiler_cfg.get_torch_profiler_config() or {}
            except Exception:
                torch_profiler_kwargs = {}

            if not torch_profiler_kwargs:
                # Minimal fallback to avoid heavy tracing overhead
                torch_profiler_kwargs = {
                    "activities": [torch.profiler.ProfilerActivity.CUDA],
                    "record_shapes": False,
                    "profile_memory": False,
                    "with_stack": False,
                    "with_flops": False,
                    "schedule": torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                }
            
            # Run benchmark with PyTorch profiler
            with torch.profiler.profile(**torch_profiler_kwargs) as prof:
                profile_iterations = min(config.iterations, 10)
                times_ms = cast(List[float], [])
                is_cuda = self.device.type == "cuda"
                
                if is_cuda:
                    # Create events once, reuse across iterations
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(self.device)  # Sync once before loop
                    
                    for _ in range(profile_iterations):
                        start_event.record()
                        fn()
                        end_event.record()
                        torch.cuda.synchronize(self.device)
                        times_ms.append(start_event.elapsed_time(end_event))
                        prof.step()  # Record each iteration in profiling trace
                else:
                    # CPU: use high-resolution timer
                    for _ in range(profile_iterations):
                        start_time = time.perf_counter()
                        fn()
                        end_time = time.perf_counter()
                        times_ms.append((end_time - start_time) * 1000)
                        prof.step()  # Record each iteration in profiling trace
            
            # Export profiling data
            trace_file = prof_dir / "trace.json"
            prof.export_chrome_trace(str(trace_file))
            profiling_outputs["pytorch_trace"] = str(trace_file)
            
            return times_ms, profiling_outputs
            
        except Exception as e:
            # Fallback to non-profiling benchmark
            times, inference_data = self._benchmark_without_profiling(fn, config)
            return times, {}
    
    
    # Profiling methods moved to profiling.profiling_runner module
    # Use run_nsys_profiling() and run_ncu_profiling() from profiling_runner.py
    # Wrapper generation moved to profiling.profiler_wrapper module
    # Use create_benchmark_wrapper() from profiler_wrapper.py
    
    def _benchmark_without_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> tuple[List[float], Optional[Dict[str, List[float]]]]:
        """Benchmark without profiling.
        
        Returns:
            Tuple of (times_ms, inference_timing_data) where inference_timing_data is None
            or a dict with 'ttft_times_ms' and 'tpot_times_ms' keys
        """
        if self.mode == BenchmarkMode.TRITON:
            times_ms = self._benchmark_triton(fn, config)
            return times_ms, None
        elif self.mode == BenchmarkMode.PYTORCH:
            times_ms = self._benchmark_pytorch(fn, config)
            return times_ms, None
        else:
            return self._benchmark_custom(fn, config)
    
    def _benchmark_triton(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use Triton's do_bench (returns single value per call)."""
        try:
            import triton.testing as tt
            times_ms = cast(List[float], [])
            # Triton do_bench handles warmup internally, but we do our own
            for _ in range(config.iterations):
                time_ms = tt.do_bench(fn, warmup=0, rep=1)  # We handle warmup
                times_ms.append(time_ms)
            return times_ms
        except ImportError:
            # Fallback to custom if Triton not available
            times, _ = self._benchmark_custom(fn, config)
            return times
    
    def _benchmark_pytorch(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use PyTorch's Timer."""
        try:
            from torch.utils.benchmark import Timer
            
            timer = Timer(
                stmt="fn()",
                globals={"fn": fn},
                num_threads=1,
            )
            
            # blocked_autorange runs until min_run_time is met
            measurement = timer.blocked_autorange(
                min_run_time=config.min_run_time_ms / 1000.0  # Convert to seconds
            )
            
            # measurement.times is already in seconds
            times_ms = [t * 1000 for t in measurement.times]
            
            # If we got fewer iterations than requested, pad with repeats
            if len(times_ms) < config.iterations:
                times_ms = (times_ms * ((config.iterations // len(times_ms)) + 1))[:config.iterations]
            
            return times_ms
        except Exception as e:
            # Fallback to custom on error
            times, _ = self._benchmark_custom(fn, config)
            return times
    
    def _benchmark_custom(self, fn: Callable, config: BenchmarkConfig) -> tuple[List[float], Optional[Dict[str, List[float]]]]:
        """Custom benchmarking with CUDA Events for accurate GPU timing.
        
        Optimized for minimal overhead:
        - Uses CUDA Events for accurate GPU timing without blocking
        - Reuses events across iterations for efficiency
        - Synchronizes only when necessary for accurate timing
        
        Returns:
            Tuple of (times_ms, inference_timing_data) where inference_timing_data is None
            or a dict with 'ttft_times_ms' and 'tpot_times_ms' keys
        """
        times_ms: List[float] = []
        inference_timing_data: Optional[Dict[str, List[float]]] = None
        ttft_times_ms: List[float] = []
        tpot_times_ms: List[float] = []

        # Some benchmarks (e.g., external CUDA binaries) report their own timing.
        benchmark_obj = getattr(fn, "__self__", None)
        use_reported_time = bool(getattr(benchmark_obj, "use_reported_time", False))
        range_name = getattr(config, "name", None) or getattr(fn, "__name__", "benchmark_fn")
        
        is_cuda = self.device.type == "cuda"
        
        if is_cuda:
            # Use CUDA Events for accurate GPU timing
            # Create events once - reuse across iterations (efficient)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Synchronize once before starting to ensure clean state
            torch.cuda.synchronize(self.device)
            
            # Run benchmark iterations with accurate per-iteration timing
            # CUDA Events provide accurate timing with minimal overhead
            for _ in range(config.iterations):
                # Record start event (non-blocking)
                start_event.record()
                # Execute function under test with NVTX
                with self._nvtx_range(range_name):
                    result = fn()
                # Record end event (non-blocking)
                end_event.record()
                # Synchronize only the end event (not device-wide) for accurate timing
                # This avoids blocking other CUDA streams and reduces overhead
                end_event.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                if use_reported_time:
                    reported = getattr(benchmark_obj, "last_time_ms", None)
                    if reported is not None:
                        elapsed_ms = reported
                times_ms.append(elapsed_ms)
                
                # Check if function returned inference timing data
                if isinstance(result, dict):
                    if "ttft_times_ms" in result and isinstance(result["ttft_times_ms"], list):
                        ttft_times_ms.extend(result["ttft_times_ms"])
                    if "tpot_times_ms" in result and isinstance(result["tpot_times_ms"], list):
                        tpot_times_ms.extend(result["tpot_times_ms"])
        else:
            # CPU: use high-resolution timer
            for _ in range(config.iterations):
                start_time = time.perf_counter()
                with self._nvtx_range(range_name):
                    result = fn()
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                if use_reported_time:
                    reported = getattr(benchmark_obj, "last_time_ms", None)
                    if reported is not None:
                        elapsed_ms = reported
                times_ms.append(elapsed_ms)
                
                # Check if function returned inference timing data
                if isinstance(result, dict):
                    if "ttft_times_ms" in result and isinstance(result["ttft_times_ms"], list):
                        ttft_times_ms.extend(result["ttft_times_ms"])
                    if "tpot_times_ms" in result and isinstance(result["tpot_times_ms"], list):
                        tpot_times_ms.extend(result["tpot_times_ms"])
        
        # Build inference timing data if collected
        if ttft_times_ms or tpot_times_ms:
            inference_timing_data = {
                "ttft_times_ms": ttft_times_ms,
                "tpot_times_ms": tpot_times_ms,
            }
        
        return times_ms, inference_timing_data
    
    def _warmup(self, fn: Callable, warmup_iterations: int) -> None:
        """Perform warmup iterations."""
        is_cuda = self.device.type == "cuda"
        range_name = getattr(fn, "__name__", "warmup")
        for _ in range(warmup_iterations):
            with self._nvtx_range(f"{range_name}_warmup"):
                fn()
        if is_cuda:
            torch.cuda.synchronize(self.device)
    
    def _compute_percentiles(self, values: List[float], percentiles: List[float]) -> Dict[float, float]:
        """Compute percentiles for a list of values.
        
        Args:
            values: List of numeric values
            percentiles: List of percentile values (e.g., [50.0, 90.0, 95.0, 99.0])
            
        Returns:
            Dictionary mapping percentile to value
        """
        if not values:
            return {}
        
        # Ensure percentiles is a list
        if percentiles is None:
            percentiles = [25, 50, 75, 99]
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        percentiles_dict = {}
        
        for p in percentiles:
            idx = int((p / 100.0) * (n - 1))
            idx = min(idx, n - 1)
            percentiles_dict[p] = sorted_values[idx]
        
        return percentiles_dict
    
    def _capture_inference_timing(
        self,
        ttft_times_ms: List[float],
        tpot_times_ms: List[float],
        config: BenchmarkConfig
    ) -> Optional[InferenceTimingStats]:
        """Capture inference timing statistics from raw TTFT and TPOT measurements.
        
        Args:
            ttft_times_ms: List of TTFT measurements (one per request) in milliseconds
            tpot_times_ms: List of TPOT measurements (one per token) in milliseconds
            config: BenchmarkConfig for percentile configuration
            
        Returns:
            InferenceTimingStats object or None if no data
        """
        if not PYDANTIC_AVAILABLE or InferenceTimingStats is None:
            return None
        
        if not ttft_times_ms or not tpot_times_ms:
            return None
        
        # Compute TTFT percentiles - ensure percentiles is a list
        percentiles = config.percentiles if config.percentiles is not None else [25, 50, 75, 99]
        ttft_percentiles = self._compute_percentiles(ttft_times_ms, percentiles)
        ttft_p50 = ttft_percentiles.get(50.0) or statistics.median(ttft_times_ms)
        ttft_p90 = ttft_percentiles.get(90.0)
        ttft_p95 = ttft_percentiles.get(95.0)
        ttft_p99 = ttft_percentiles.get(99.0)
        
        # Compute TPOT percentiles
        tpot_percentiles = self._compute_percentiles(tpot_times_ms, percentiles)
        tpot_p50 = tpot_percentiles.get(50.0) or statistics.median(tpot_times_ms)
        tpot_p90 = tpot_percentiles.get(90.0)
        tpot_p95 = tpot_percentiles.get(95.0)
        tpot_p99 = tpot_percentiles.get(99.0)
        
        # Count requests and tokens
        num_requests = len(ttft_times_ms)
        total_tokens_generated = len(tpot_times_ms)
        
        return InferenceTimingStats(
            ttft_mean_ms=statistics.mean(ttft_times_ms),
            ttft_p50_ms=ttft_p50,
            ttft_p90_ms=ttft_p90,
            ttft_p95_ms=ttft_p95,
            ttft_p99_ms=ttft_p99,
            ttft_percentiles=ttft_percentiles,
            tpot_mean_ms=statistics.mean(tpot_times_ms),
            tpot_p50_ms=tpot_p50,
            tpot_p90_ms=tpot_p90,
            tpot_p95_ms=tpot_p95,
            tpot_p99_ms=tpot_p99,
            tpot_percentiles=tpot_percentiles,
            num_requests=num_requests,
            total_tokens_generated=total_tokens_generated,
            raw_ttft_times_ms=ttft_times_ms,
            raw_tpot_times_ms=tpot_times_ms,
            schemaVersion="1.0",
        )
    
    def _compute_stats(
        self, times_ms: List[float], config: BenchmarkConfig
    ) -> PydanticBenchmarkResult:
        """Compute statistical measures and return Pydantic BenchmarkResult."""
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required. Install with: pip install pydantic")
        
        if not times_ms:
            raise ValueError("No timing data collected")
        
        # CRITICAL: Filter out None values and ensure all values are numeric
        # This prevents "'<' not supported between instances of 'NoneType' and 'list'" errors
        valid_times_ms: List[float] = []
        for t in times_ms:
            if t is not None and isinstance(t, (int, float)):
                valid_times_ms.append(float(t))
            elif isinstance(t, list):
                # If we somehow got a list, extract numeric values from it
                valid_times_ms.extend([float(x) for x in t if x is not None and isinstance(x, (int, float))])
        
        if not valid_times_ms:
            raise ValueError(f"No valid timing data collected. Original times_ms had {len(times_ms)} entries, but none were valid numbers.")
        
        sorted_times = sorted(valid_times_ms)
        n = len(sorted_times)
        
        # Compute percentiles - ensure percentiles is a list
        percentiles = config.percentiles if config.percentiles is not None else [25, 50, 75, 99]
        percentiles_dict = {}
        for p in percentiles:
            idx = int((p / 100.0) * (n - 1))
            idx = min(idx, n - 1)
            percentiles_dict[p] = sorted_times[idx]
        
        # Extract p50/p90/p95/p99 from percentiles
        p50 = percentiles_dict.get(50.0) or statistics.median(valid_times_ms)
        p90 = percentiles_dict.get(90.0)
        p95 = percentiles_dict.get(95.0)
        p99 = percentiles_dict.get(99.0)
        
        timing = TimingStats(
            mean_ms=statistics.mean(valid_times_ms),
            median_ms=statistics.median(valid_times_ms),
            std_ms=statistics.stdev(valid_times_ms) if n > 1 else 0.0,
            min_ms=min(valid_times_ms),
            max_ms=max(valid_times_ms),
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            percentiles=percentiles_dict,
            iterations=n,
            warmup_iterations=config.warmup,
            raw_times_ms=valid_times_ms,  # Store cleaned data, not original
            schemaVersion="1.0",
        )
        
        device_index = None
        if torch.cuda.is_available():
            device_index = self.device.index if getattr(self.device, "index", None) is not None else torch.cuda.current_device()
        gpu_metrics = query_gpu_telemetry(device_index=device_index)

        result = PydanticBenchmarkResult(
            timing=timing,
            inference_timing=None,
            memory=None,
            artifacts=None,
            profiler_metrics=None,
            errors=[],
            validation_status=None,
            validation_message=None,
            benchmark_name=None,
            device=str(self.device),
            mode=self.mode.value,
            timeout_stage=None,
            timeout_duration_seconds=None,
            timeout_limit_seconds=None,
            gpu_metrics=gpu_metrics,
            schemaVersion="1.0",
        )
        self._annotate_launch_metadata(result, config)
        return result

    def _resolve_workload_metadata(self, benchmark: BaseBenchmark) -> Optional[WorkloadMetadata]:
        """Resolve workload metadata declared by the benchmark (if any)."""
        metadata: Optional[WorkloadMetadata] = None
        getter = getattr(benchmark, "get_workload_metadata", None)
        if callable(getter):
            try:
                metadata = getter()
            except Exception as exc:  # pragma: no cover - defensive
                if LOGGER_AVAILABLE:
                    logger.debug(f"get_workload_metadata() raised: {exc}")
        if metadata is None:
            candidate = getattr(benchmark, "workload_metadata", None)
            if isinstance(candidate, WorkloadMetadata):
                metadata = candidate
        if metadata is None:
            metadata = self._infer_workload_metadata_from_attributes(benchmark)
        return metadata

    def _resolve_custom_metrics(self, benchmark: BaseBenchmark) -> Optional[Dict[str, float]]:
        """Resolve benchmark-specific custom metrics if provided."""
        getter = getattr(benchmark, "get_custom_metrics", None)
        if callable(getter):
            try:
                metrics = getter()
                if isinstance(metrics, dict) and metrics:
                    numeric_metrics: Dict[str, float] = {}
                    for key, value in metrics.items():
                        if isinstance(value, bool):
                            numeric_metrics[key] = float(value)
                        elif isinstance(value, (int, float)):
                            numeric_metrics[key] = float(value)
                    if numeric_metrics:
                        return numeric_metrics
            except Exception as exc:  # pragma: no cover - defensive
                if LOGGER_AVAILABLE:
                    logger.debug(f"get_custom_metrics() raised: {exc}")
        return None

    def _infer_workload_metadata_from_attributes(self, benchmark: BaseBenchmark) -> Optional[WorkloadMetadata]:
        """Best-effort inference of workload metadata from common benchmark attributes."""
        def _read_numeric_attr(names: List[str]) -> Optional[float]:
            for name in names:
                if hasattr(benchmark, name):
                    value = getattr(benchmark, name)
                    if isinstance(value, (int, float)) and value > 0:
                        return float(value)
            return None

        batch = _read_numeric_attr([
            "batch_size",
            "train_batch_size",
            "eval_batch_size",
            "micro_batch_size",
            "micro_batchsize",
        ])
        seq_len = _read_numeric_attr([
            "seq_len",
            "seq_length",
            "sequence_length",
            "token_count",
            "tokens",
        ])
        explicit_tokens = _read_numeric_attr([
            "tokens_per_iteration",
            "tokens_per_step",
            "tokens_per_batch",
        ])
        explicit_requests = _read_numeric_attr([
            "requests_per_iteration",
            "requests_per_step",
            "num_requests",
        ])
        samples = _read_numeric_attr([
            "samples_per_iteration",
            "samples_per_step",
            "num_samples",
        ])
        bytes_per_iter = _read_numeric_attr([
            "bytes_per_iteration",
            "payload_bytes",
            "data_bytes",
        ])
        custom_units = _read_numeric_attr([
            "workload_units_per_iteration",
            "flops_per_iteration",
        ])
        goodput = _read_numeric_attr([
            "target_goodput",
            "expected_goodput",
        ])

        tokens = explicit_tokens
        if tokens is None:
            if batch is not None and seq_len is not None:
                tokens = batch * seq_len
            elif seq_len is not None:
                tokens = seq_len
            elif batch is not None:
                tokens = batch

        if samples is None and batch is not None:
            samples = batch

        if custom_units is not None and getattr(benchmark, "workload_unit_name", None):
            custom_unit_name = getattr(benchmark, "workload_unit_name")
        else:
            custom_unit_name = None

        requests = explicit_requests or batch or 1.0

        if (
            tokens is None
            and samples is None
            and bytes_per_iter is None
            and custom_units is None
            and (requests == 1.0 or requests is None)
        ):
            return None

        return WorkloadMetadata(
            requests_per_iteration=requests or 1.0,
            tokens_per_iteration=tokens,
            samples_per_iteration=samples,
            bytes_per_iteration=bytes_per_iter,
            custom_units_per_iteration=custom_units,
            custom_unit_name=custom_unit_name,
            goodput=goodput,
        )

    def _compute_throughput_stats(
        self,
        timing: Optional[TimingStats],
        workload: Optional[WorkloadMetadata],
    ) -> Optional[ThroughputStats]:
        """Convert timing + workload metadata into throughput statistics."""
        if timing is None or timing.mean_ms <= 0:
            return None
        mean_seconds = timing.mean_ms / 1000.0
        requests_per_iteration = 1.0
        tokens_per_iteration = None
        samples_per_iteration = None
        bytes_per_iteration = None
        custom_units_per_iteration = None
        custom_unit_name = None
        goodput = None
        
        if workload is not None:
            if workload.requests_per_iteration is not None:
                requests_per_iteration = max(float(workload.requests_per_iteration), 0.0)
            if workload.tokens_per_iteration is not None:
                tokens_per_iteration = float(workload.tokens_per_iteration)
            if workload.samples_per_iteration is not None:
                samples_per_iteration = float(workload.samples_per_iteration)
            if workload.bytes_per_iteration is not None:
                bytes_per_iteration = float(workload.bytes_per_iteration)
            if workload.custom_units_per_iteration is not None:
                custom_units_per_iteration = float(workload.custom_units_per_iteration)
            custom_unit_name = workload.custom_unit_name
            goodput = workload.goodput
        
        requests_per_s = requests_per_iteration / mean_seconds if mean_seconds > 0 else None
        tokens_per_s = (
            tokens_per_iteration / mean_seconds if tokens_per_iteration is not None and mean_seconds > 0 else None
        )
        samples_per_s = (
            samples_per_iteration / mean_seconds if samples_per_iteration is not None and mean_seconds > 0 else None
        )
        bytes_per_s = (
            bytes_per_iteration / mean_seconds if bytes_per_iteration is not None and mean_seconds > 0 else None
        )
        custom_unit_per_s = (
            custom_units_per_iteration / mean_seconds
            if custom_units_per_iteration is not None and mean_seconds > 0
            else None
        )
        
        return ThroughputStats(
            requests_per_s=requests_per_s,
            tokens_per_s=tokens_per_s,
            samples_per_s=samples_per_s,
            bytes_per_s=bytes_per_s,
            custom_unit_per_s=custom_unit_per_s,
            custom_unit_name=custom_unit_name,
            latency_ms=timing.mean_ms,
            goodput=goodput,
        )

    def _attach_throughput_metrics(self, result: PydanticBenchmarkResult, benchmark: BaseBenchmark) -> None:
        """Attach throughput metrics to the benchmark result when possible."""
        workload = self._resolve_workload_metadata(benchmark)
        throughput_stats = self._compute_throughput_stats(result.timing, workload)
        if throughput_stats:
            result.throughput = throughput_stats


def compare_benchmarks(
    baseline: BaseBenchmark,
    optimized: BaseBenchmark,
    harness: Optional[BenchmarkHarness] = None,
    name: str = "Comparison",
    regression_threshold_pct: float = 5.0
) -> Dict[str, Any]:
    """Compare baseline vs optimized benchmarks and return metrics.
    
    Args:
        baseline: Baseline benchmark instance
        optimized: Optimized benchmark instance
        harness: BenchmarkHarness instance (creates new if None)
        name: Name for the comparison
        regression_threshold_pct: Percentage degradation to consider a regression (default: 5%)
        
    Returns:
        Dictionary with comparison metrics including regression detection
    """
    if harness is None:
        harness = BenchmarkHarness()
    
    baseline_result = harness.benchmark(baseline)
    optimized_result = harness.benchmark(optimized)
    
    baseline_mean = baseline_result.timing.mean_ms if baseline_result.timing else 0.0
    optimized_mean = optimized_result.timing.mean_ms if optimized_result.timing else 0.0
    speedup = baseline_mean / optimized_mean if optimized_mean > 0 else 1.0
    
    # Detect regression: optimized is slower by threshold
    regression = False
    regression_pct = None
    if speedup < 1.0:
        regression_pct = (1.0 - speedup) * 100
        regression = regression_pct >= regression_threshold_pct
    
    return {
        "name": name,
        "baseline": {
            "mean_ms": baseline_result.timing.mean_ms if baseline_result.timing else 0.0,
            "median_ms": baseline_result.timing.median_ms if baseline_result.timing else 0.0,
            "std_ms": baseline_result.timing.std_ms if baseline_result.timing else 0.0,
            "min_ms": baseline_result.timing.min_ms if baseline_result.timing else 0.0,
            "max_ms": baseline_result.timing.max_ms if baseline_result.timing else 0.0,
        },
        "optimized": {
            "mean_ms": optimized_result.timing.mean_ms if optimized_result.timing else 0.0,
            "median_ms": optimized_result.timing.median_ms if optimized_result.timing else 0.0,
            "std_ms": optimized_result.timing.std_ms if optimized_result.timing else 0.0,
            "min_ms": optimized_result.timing.min_ms if optimized_result.timing else 0.0,
            "max_ms": optimized_result.timing.max_ms if optimized_result.timing else 0.0,
        },
        "speedup": speedup,
        "regression": regression,
        "regression_pct": regression_pct,
        "baseline_result": baseline_result,
        "optimized_result": optimized_result,
    }
