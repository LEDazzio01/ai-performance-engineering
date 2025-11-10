#!/usr/bin/env python3
"""Run every single benchmark/example and summarize results.

This script:
1. Discovers all baseline/optimized pairs across all chapters
2. Runs actual benchmarks using BenchmarkHarness
3. Collects performance metrics (speedup, latency, throughput, etc.)
4. Generates a comprehensive summary report

Usage:
    python benchmark.py [--chapter ch1|all] [--format json|markdown|both]
"""

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
import sys
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import defaultdict
import statistics
from dataclasses import dataclass

# Ensure repository root on sys.path before importing helpers
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.env_defaults import apply_env_defaults, dump_environment_and_capabilities

apply_env_defaults()

import torch
import subprocess
import time
import os
import tempfile
from typing import List, Tuple, Any
from common.python.chapter_compare_template import discover_benchmarks, load_benchmark
from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig
from common.python.run_manifest import reset_gpu_state
try:
    from common.python.cuda_binary_benchmark import detect_supported_arch
except ImportError:  # pragma: no cover - optional dependency during docs builds
    detect_supported_arch = None  # type: ignore[assignment]

# Import logger
try:
    from common.python.logger import get_logger, setup_logging
    logger = get_logger(__name__)
    LOGGER_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGGER_AVAILABLE = False
    def setup_logging(*args, **kwargs):
        pass

# Check if torch.profiler is available at module level
TORCH_PROFILER_AVAILABLE = hasattr(torch, 'profiler') and hasattr(torch.profiler, 'profile')

# Import metric extraction utilities
try:
    from tools.analysis.metric_extractor import (
        extract_from_ncu_report,
        extract_from_nsys_report,
    )
except ImportError:
    # Fallback if metric extractor not available
    def extract_from_ncu_report(path: Path) -> Dict[str, float]:
        return {}
    def extract_from_nsys_report(path: Path) -> Dict[str, float]:
        return {}


def reset_gpu_via_script(reason: str) -> None:
    """Invoke tools/reset_gpu.py with the provided reason."""
    reset_script = Path(__file__).resolve().parents[1] / "tools" / "reset_gpu.py"
    if not reset_script.exists():
        logger.warning("GPU reset script not found at %s", reset_script)
        return
    try:
        subprocess.run(
            [sys.executable, str(reset_script), "--reason", reason],
            check=False,
            timeout=180,
        )
    except Exception as exc:
        logger.warning("GPU reset script failed: %s", exc)


def maybe_reset_gpu_for_error(error_message: str, context: str) -> None:
    """Trigger GPU reset when the error message indicates a hang/timeout."""
    normalized = error_message.strip().upper()
    if "TIMEOUT" in normalized or "HANG" in normalized:
        reset_gpu_via_script(f"{context}: {error_message.splitlines()[0]}")


def extract_from_pytorch_trace(trace_path: Path) -> Dict[str, float]:
    """Extract metrics from PyTorch Chrome trace JSON file.
    
    Args:
        trace_path: Path to Chrome trace JSON file
        
    Returns:
        Dictionary of extracted metrics
    """
    if not trace_path.exists():
        return {}
    
    metrics = {}
    
    try:
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)
        
        # Chrome trace format: {"traceEvents": [...], "displayTimeUnit": "ms"}
        if isinstance(trace_data, dict) and "traceEvents" in trace_data:
            events = trace_data["traceEvents"]
            
            # Sum CUDA kernel times
            cuda_time_us = 0
            cpu_time_us = 0
            cuda_kernels = 0
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                # Look for CUDA kernel events
                if event.get("cat") == "cuda_runtime" or "cuda" in event.get("name", "").lower():
                    dur = event.get("dur", 0)  # Duration in microseconds
                    if dur > 0:
                        cuda_time_us += dur
                        cuda_kernels += 1
                
                # Look for CPU events
                if event.get("cat") == "cpu_op" or "cpu" in event.get("cat", "").lower():
                    dur = event.get("dur", 0)
                    if dur > 0:
                        cpu_time_us += dur
            
            if cuda_time_us > 0:
                metrics["pytorch_cuda_time_us"] = cuda_time_us
                metrics["pytorch_cuda_time_ms"] = cuda_time_us / 1000.0
            if cpu_time_us > 0:
                metrics["pytorch_cpu_time_us"] = cpu_time_us
                metrics["pytorch_cpu_time_ms"] = cpu_time_us / 1000.0
            if cuda_kernels > 0:
                metrics["pytorch_cuda_kernels"] = float(cuda_kernels)
                
    except Exception:
        pass
    
    return metrics


PROOF_OF_BENEFIT_REQUIREMENTS: Dict[str, Dict[str, float]] = {
    "ch20": {
        "integrated": 1.05,
        "kv": 1.05,
        "pipeline": 1.05,
        "precision": 1.05,
        "training": 1.05,
    },
    "capstone": {
        "01_system_foundations": 1.05,
        "02_cluster_parallelism": 1.05,
        "03_io_pipeline": 1.05,
        "04_kernel_optimization": 1.05,
        "05_streams_and_graphs": 1.05,
        "06_compiler_stack": 1.05,
        "07_inference_attention": 1.05,
        "08_low_precision": 1.05,
        "09_end_to_end": 1.05,
    },
}

INFORMATIONAL_BENCHMARKS: Dict[str, Set[str]] = {}

SMOKE_CUDA_ITERATION_OVERRIDES: Dict[str, int] = {
    # CUDA microbenchmarks in these chapters are highly jittery; give them more samples even in smoke mode.
    "ch6": 8,
    "ch7": 12,
    "ch10": 10,
    "ch11": 8,
    "ch12": 10,
}


def format_time_ms(time_ms: float) -> str:
    """Format time in milliseconds with adaptive precision.
    
    For very small values (< 1ms), use more decimal places to show actual timing.
    For larger values, use 2 decimal places.
    Handles zero and negative values appropriately.
    """
    if time_ms <= 0.0:
        return f"{time_ms:.2f}"
    elif time_ms < 0.001:
        return f"{time_ms:.6f}"  # microseconds precision
    elif time_ms < 0.01:
        return f"{time_ms:.5f}"
    elif time_ms < 0.1:
        return f"{time_ms:.4f}"
    elif time_ms < 1.0:
        return f"{time_ms:.3f}"
    else:
        return f"{time_ms:.2f}"


def format_throughput_summary(throughput_obj: Optional[Any]) -> str:
    """Pretty print throughput metrics for logs."""
    if throughput_obj is None:
        return ""
    parts: List[str] = []
    requests = getattr(throughput_obj, "requests_per_s", None)
    tokens = getattr(throughput_obj, "tokens_per_s", None)
    samples = getattr(throughput_obj, "samples_per_s", None)
    latency_ms = getattr(throughput_obj, "latency_ms", None)
    goodput = getattr(throughput_obj, "goodput", None)
    if requests:
        parts.append(f"{requests:,.2f} req/s")
    if tokens:
        parts.append(f"{tokens:,.2f} tokens/s")
    if samples and samples != tokens:
        parts.append(f"{samples:,.2f} samples/s")
    if latency_ms:
        parts.append(f"{latency_ms:.3f} ms/iter")
    if goodput is not None:
        parts.append(f"goodput={goodput:.2%}")
    return ", ".join(parts)


def serialize_throughput(throughput_obj: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Convert ThroughputStats into a JSON-serializable dict."""
    if throughput_obj is None:
        return None
    if hasattr(throughput_obj, "model_dump"):
        return throughput_obj.model_dump()
    if hasattr(throughput_obj, "__dict__"):
        return dict(throughput_obj.__dict__)
    return None


def reset_cuda_state():
    """Reset CUDA state to prevent cascading failures."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass
    except Exception:
        pass


def is_distributed_benchmark(file_path: Path) -> bool:
    """Check if a benchmark file contains distributed operations.
    
    This function detects distributed benchmarks by looking for:
    - torch.distributed imports and usage
    - DistributedDataParallel (DDP)
    - NCCL backend usage
    - Environment variables like WORLD_SIZE, RANK
    - Multi-GPU communication patterns
    """
    try:
        content = file_path.read_text()
        
        # Check for distributed imports
        has_dist_import = any(pattern in content for pattern in [
            'import torch.distributed',
            'from torch.distributed',
            'torch.distributed as dist',
        ])
        
        # Check for distributed operations
        has_dist_ops = any(pattern in content for pattern in [
            'dist.init_process_group',
            'torch.distributed.init_process_group',
            'torch.nn.parallel.DistributedDataParallel',
            'DistributedDataParallel(',
            'DDP(',
        ])
        
        # Check for NCCL backend (strong indicator of multi-GPU)
        has_nccl = any(pattern in content for pattern in [
            "backend='nccl'",
            'backend="nccl"',
            'backend = "nccl"',
            'backend = \'nccl\'',
        ])
        
        # Check for distributed environment variables (but not just setup code)
        # Only count if it's actually used, not just set
        has_world_size = 'WORLD_SIZE' in content and ('os.environ' in content or 'getenv' in content)
        has_rank = 'RANK' in content and ('os.environ' in content or 'getenv' in content)
        
        # A benchmark is distributed if it has distributed imports AND operations
        # OR if it explicitly uses NCCL backend
        return (has_dist_import and has_dist_ops) or has_nccl or (has_world_size and has_rank and has_dist_ops)
    except Exception:
        return False


def check_hardware_limitation(error_msg: str) -> Optional[str]:
    """Check if error is due to hardware/software limitation and return skip reason.
    
    Only skips for TRUE hardware limitations that cannot be fixed:
    - Triton SM 12.1 bug (sm_121a issue)
    
    For other issues, we should fix them instead of skipping:
    - CUTLASS: Verify it's actually unavailable before skipping
    - CUDA extensions: Should be pre-compiled, not skipped
    - torch.compile timeouts: Should reduce model size, not skip
    - Device-side asserts: Already handled with reset_cuda_state()
    """
    error_lower = error_msg.lower()
    
    # Treat explicit SKIPPED markers as hardware limitations.
    if 'skipped:' in error_lower:
        reason = error_msg[error_lower.find('skipped:') + len('skipped:'):].strip()
        return reason or "Benchmark reported SKIPPED"
    
    # Device-side assert cascades - these should be prevented by reset_cuda_state()
    # But if they still occur, it's a transient state issue, not a hardware limitation
    if 'device-side assert' in error_lower or 'cudaerrorassert' in error_lower:
        # Don't skip - reset should handle this. Return None to let it fail normally.
        return None
    
    # CUDA pipeline API unavailable (older GPUs)
    if 'cuda pipeline api unavailable' in error_lower:
        return "CUDA Pipeline API not supported on this GPU"
    if 'requires compute capability >= 8.0' in error_lower and 'pipeline' in error_lower:
        return "CUDA Pipeline API requires compute capability >= 8.0"
    
    if 'distributed shared memory unavailable' in error_lower:
        return "Distributed shared memory (DSMEM) not enabled on this hardware/driver"
    if 'cuda 13+ required for cluster dsmem support' in error_lower:
        return "Distributed shared memory (DSMEM) not supported by this toolkit/runtime"
    if 'thread block clusters unavailable' in error_lower or 'cluster target block not present' in error_lower:
        return "Thread block clusters unavailable on this driver/toolkit (needs CUDA 13.1+ or compute-sanitizer)"
    if 'distributed benchmark requires multiple gpus' in error_lower:
        if 'SKIPPED:' in error_msg:
            return error_msg.split('SKIPPED:', 1)[1].strip()
        return "Distributed benchmark requires multiple GPUs (insufficient GPUs available)"
    
    # Segmentation faults - these should be prevented by pre-compilation
    # If they still occur after pre-compilation, it's a real issue, not a limitation
    if 'segmentation fault' in error_lower or 'segfault' in error_lower or 'sigsegv' in error_lower:
        # Don't skip - extensions should be pre-compiled
        return None
    
    # CUTLASS backend - verify it's actually unavailable before skipping
    if 'cutlass' in error_lower and ('attributeerror' in error_lower or 'loweringexception' in error_lower):
        # Check if CUTLASS is actually installed
        try:
            import cutlass
            import importlib_metadata
            try:
                version = importlib_metadata.version("nvidia-cutlass-dsl")
                # CUTLASS is installed - this might be a configuration issue, not unavailability
                # Don't skip - let it fail with clear error message
                return None
            except importlib_metadata.PackageNotFoundError:
                # CUTLASS package not found - might be truly unavailable
                pass
        except ImportError:
            # CUTLASS not installed - might be truly unavailable
            pass
        # Only skip if we're sure CUTLASS is not available
        # For now, don't skip - let the fallback logic handle it
        return None
    
    # CUDA extension failures - should be pre-compiled, not skipped
    if 'cuda extension' in error_lower or 'failed to load/compile' in error_lower:
        # Don't skip - extensions should be pre-compiled before running tests
        return None
    
    # TF32 API mixing - this is a code issue, not a hardware limitation
    if 'mix of the legacy and new apis' in error_lower or 'allow_tf32_new' in error_lower:
        # Don't skip - this should be fixed in arch_config.py
        return None
    
    return None


def discover_cuda_benchmarks(chapter_dir: Path) -> List[Tuple[Path, List[Path], str]]:
    """Discover CUDA benchmark pairs by looking for baseline_*.cu files with matching optimized_*.cu.
    
    Uses precise matching: each optimized file matches the most specific baseline (longest matching prefix).
    This prevents multiple baselines from matching the same optimized files.
    
    Args:
        chapter_dir: Path to chapter directory (e.g., Path('ch1'))
        
    Returns:
        List of tuples: (baseline_cu_path, [optimized_cu_paths], example_name)
        Example: (Path('ch1/baseline_gemm.cu'), [Path('ch1/optimized_gemm_batched.cu')], 'gemm')
    """
    baseline_files = sorted(chapter_dir.glob("baseline_*.cu"), key=lambda p: len(p.stem), reverse=True)
    all_optimized_files = list(chapter_dir.glob("optimized_*.cu"))
    
    # Map each optimized file to its most specific baseline
    optimized_to_baseline = {}  # optimized_path -> baseline_path
    
    for baseline_file in baseline_files:
        baseline_name = baseline_file.stem  # e.g., "baseline_gemm" or "baseline_cuda_graphs_conditional"
        baseline_suffix = baseline_name.replace("baseline_", "")  # e.g., "gemm" or "cuda_graphs_conditional"
        baseline_prefix = f"optimized_{baseline_suffix}"
        
        # Find optimized files that match this baseline
        for opt_file in all_optimized_files:
            opt_stem = opt_file.stem  # e.g., "optimized_gemm_batched" or "optimized_cuda_graphs_conditional"
            
            # Match if optimized file starts with baseline_prefix followed by end of string or underscore
            if opt_stem == baseline_prefix or opt_stem.startswith(baseline_prefix + "_"):
                # Only assign if not already assigned to a more specific baseline
                if opt_file not in optimized_to_baseline:
                    optimized_to_baseline[opt_file] = baseline_file
    
    # Group optimized files by baseline
    baseline_to_optimized = {}  # baseline_path -> [optimized_paths]
    for opt_file, baseline_file in optimized_to_baseline.items():
        if baseline_file not in baseline_to_optimized:
            baseline_to_optimized[baseline_file] = []
        baseline_to_optimized[baseline_file].append(opt_file)
    
    # Build pairs
    pairs = []
    for baseline_file, optimized_files in baseline_to_optimized.items():
        baseline_suffix = baseline_file.stem.replace("baseline_", "")
        example_name = baseline_suffix.split("_")[0]
        pairs.append((baseline_file, sorted(optimized_files), example_name))
    
    return pairs


def cuda_binary_requires_multi_gpu(path: Path) -> bool:
    """Best-effort heuristic to detect CUDA binaries that require multi-GPU hardware."""
    name = path.stem.lower()
    multi_gpu_tokens = ("nvlink", "multigpu", "multi_gpu", "multi-gpu", "distributed")
    return any(token in name for token in multi_gpu_tokens)


def find_cuda_executable(cu_file: Path, chapter_dir: Path) -> Optional[Path]:
    """Find the compiled executable for a CUDA source file.
    
    Looks for executables with SM suffixes (e.g., baseline_gemm_sm121) or without suffix.
    
    Args:
        cu_file: Path to .cu source file
        chapter_dir: Path to chapter directory (for Makefile detection)
        
    Returns:
        Path to executable if found, None otherwise
    """
    base_name = cu_file.stem
    
    # Check common SM suffixes (in order of preference)
    suffixes = ["_sm121", "_sm103", "_sm100", "_sm90", "_sm89", "_sm86", ""]
    
    for suffix in suffixes:
        executable = chapter_dir / f"{base_name}{suffix}"
        if executable.exists() and os.access(executable, os.X_OK):
            return executable
    
    return None


@dataclass
class CudaBenchmarkResult:
    """Statistical results from CUDA executable benchmarking."""
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    percentiles: Dict[float, float]  # e.g., {25.0: 1.23, 50.0: 1.45, ...}
    iterations: int
    warmup_iterations: int
    skip_reason: Optional[str] = None


def benchmark_cuda_executable(executable: Path, iterations: int = 3, warmup: int = 0, timeout: int = 15) -> Optional[CudaBenchmarkResult]:
    """Benchmark a CUDA executable and return statistical results.
    
    Args:
        executable: Path to CUDA executable
        iterations: Number of benchmark iterations
        warmup: Number of warmup runs
        timeout: Timeout per run in seconds (default: 15 seconds to prevent hangs)
        
    Returns:
        CudaBenchmarkResult with statistical measures, or None if failed
    """
    import os
    import signal
    
    times_ms = []
    skip_reason: Optional[str] = None
    SKIP_EXIT_CODES = {3}
    
    def decode_message(stdout: bytes, stderr: bytes, returncode: int) -> str:
        for stream in (stderr, stdout):
            if stream:
                text = stream.decode('utf-8', errors='ignore').strip()
                if text:
                    return text.splitlines()[0]
        return f"{executable.name} exited with code {returncode}"
    
    def safe_kill_process(process):
        """Safely kill a process and its children."""
        try:
            # Try to kill process group if setsid was used
            if hasattr(os, 'setsid'):
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        process.wait()
                except (ProcessLookupError, OSError, AttributeError):
                    # Fallback to killing just the process
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            else:
                # No setsid support, just kill the process
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except (ProcessLookupError, OSError):
            # Process already terminated
            pass
    
    # Warmup runs (executables already perform their own warmup/averaging internally)
    for _ in range(warmup):
        try:
            # Run executable with timeout protection
            process = subprocess.Popen(
                [str(executable)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group if available
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                if process.returncode in SKIP_EXIT_CODES:
                    reason = decode_message(stdout, stderr, process.returncode)
                    skip_reason = reason or "Hardware/software limitation detected"
                    logger.warning(f"CUDA executable {executable.name} reported SKIPPED during warmup: {skip_reason}")
                    return CudaBenchmarkResult(
                        mean_ms=0.0,
                        median_ms=0.0,
                        std_ms=0.0,
                        min_ms=0.0,
                        max_ms=0.0,
                        percentiles={},
                        iterations=0,
                        warmup_iterations=warmup,
                        skip_reason=skip_reason,
                    )
                if process.returncode != 0:
                    # Executable failed, skip warmup
                    continue
            except subprocess.TimeoutExpired:
                # Timeout occurred - kill the process
                safe_kill_process(process)
                logger.warning(f"CUDA executable {executable.name} timed out during warmup (>{timeout}s)")
                reset_gpu_via_script(f"{executable.name} warmup timeout")
                return None
        except Exception as e:
            # If process creation failed, return None
            logger.warning(f"Failed to run CUDA executable {executable.name}: {e}")
            return None
    
    # Benchmark runs
    # Ensure CUDA libraries are in library path
    env = os.environ.copy()
    cuda_lib_paths = [
        '/usr/lib/aarch64-linux-gnu',  # libcuda.so.1 location
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda-13.0/targets/sbsa-linux/lib',
    ]
    existing_ld_path = env.get('LD_LIBRARY_PATH', '')
    new_ld_path = ':'.join(cuda_lib_paths + ([existing_ld_path] if existing_ld_path else []))
    env['LD_LIBRARY_PATH'] = new_ld_path
    
    for i in range(iterations):
        try:
            start = time.perf_counter()
            # Run executable with timeout protection
            process = subprocess.Popen(
                [str(executable)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group if available
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                end = time.perf_counter()
                
                if process.returncode == 0:
                    elapsed_ms = (end - start) * 1000.0
                    times_ms.append(elapsed_ms)
                elif process.returncode in SKIP_EXIT_CODES:
                    skip_reason = decode_message(stdout, stderr, process.returncode)
                    logger.warning(
                        f"CUDA executable {executable.name} reported SKIPPED on iteration {i+1}: {skip_reason}"
                    )
                    break
                else:
                    # Executable failed, log but continue
                    logger.warning(f"CUDA executable {executable.name} failed with return code {process.returncode} on iteration {i+1}")
                    if stderr:
                        logger.warning(f"  stderr: {stderr.decode('utf-8', errors='ignore')[:200]}")
            except subprocess.TimeoutExpired:
                # Timeout occurred - kill the process
                safe_kill_process(process)
                logger.warning(f"CUDA executable {executable.name} timed out on iteration {i+1} (>{timeout}s)")
                reset_gpu_via_script(f"{executable.name} measurement timeout")
                return None
        except Exception as e:
            # If process creation failed, log and return None
            logger.warning(f"Failed to run CUDA executable {executable.name} on iteration {i+1}: {e}")
            return None
    
    if skip_reason:
        return CudaBenchmarkResult(
            mean_ms=0.0,
            median_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            percentiles={},
            iterations=0,
            warmup_iterations=warmup,
            skip_reason=skip_reason,
        )
    
    if not times_ms:
        return None
    
    # Compute statistics similar to BenchmarkHarness._compute_stats
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    
    # Compute percentiles (same as BenchmarkHarness)
    # Use float keys to match how they're accessed (99.0, 75.0, etc.)
    percentiles_to_compute = [25.0, 50.0, 75.0, 99.0]
    percentiles_dict = {}
    for p in percentiles_to_compute:
        idx = int((p / 100.0) * (n - 1))
        idx = min(idx, n - 1)
        percentiles_dict[p] = sorted_times[idx]
    
    return CudaBenchmarkResult(
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        std_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        percentiles=percentiles_dict,
        iterations=n,
        warmup_iterations=warmup,
    )


def check_nsys_available() -> bool:
    """Check if nsys is available on the system."""
    try:
        result = subprocess.run(
            ["nsys", "--version"],
            capture_output=True,
            timeout=5,
            check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_ncu_available() -> bool:
    """Check if ncu (NVIDIA Compute Profiler) is available on the system."""
    try:
        result = subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            timeout=5,
            check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def profile_python_benchmark(
    benchmark: Any,  # Benchmark instance
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> Optional[Path]:
    """Profile a Python benchmark using nsys by wrapping benchmark execution.
    
    Args:
        benchmark: Benchmark instance (already loaded)
        benchmark_path: Path to Python benchmark file (for naming)
        chapter_dir: Path to chapter directory
        output_dir: Directory to save nsys-rep file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated nsys-rep file, or None if failed
    """
    if not check_nsys_available():
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on benchmark name
    benchmark_name = benchmark_path.stem
    nsys_output = output_dir / f"{benchmark_name}_{variant}.nsys-rep"
    
    # Create a temporary wrapper script that runs the benchmark
    wrapper_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    
    try:
        wrapper_script.write(f"""
import sys
from pathlib import Path

# Add chapter directory to path
sys.path.insert(0, r'{chapter_dir}')

# Import and load benchmark
from {benchmark_path.stem} import get_benchmark

benchmark = get_benchmark()
benchmark.setup()

# Warmup
for _ in range(5):
    benchmark.benchmark_fn()

# Profile execution
import torch
if torch.cuda.is_available():
    torch.cuda.synchronize()

benchmark.benchmark_fn()

if torch.cuda.is_available():
    torch.cuda.synchronize()

benchmark.teardown()
""")
        wrapper_script.close()
        
        # Build nsys command
        nsys_command = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "-o",
            str(nsys_output.with_suffix("")),  # nsys adds .nsys-rep automatically
            "-t", "cuda,nvtx,osrt,cublas,cudnn",
            "-s", "cpu",
            "--python-sampling=true",
            "--python-sampling-frequency=1000",
            "--cudabacktrace=true",
            "--stats=true",
            sys.executable,
            wrapper_script.name
        ]
        
        # nsys profiling timeout: 120 seconds (matches benchmark_harness.nsys_timeout_seconds)
        # nsys needs time to initialize, run benchmark (up to 15s), and collect profiling data
        result = subprocess.run(
            nsys_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=120,  # Increased from 15s - nsys profiling needs more time
            check=False
        )
        
        # Clean up wrapper script
        try:
            Path(wrapper_script.name).unlink()
        except Exception:
            pass
        
        if result.returncode == 0 and nsys_output.exists():
            return nsys_output
        else:
            return None
    except Exception:
        # Clean up wrapper script on error
        try:
            Path(wrapper_script.name).unlink()
        except Exception:
            pass
        return None


def profile_cuda_executable(
    executable: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> Optional[Path]:
    """Profile a CUDA executable using nsys.
    
    Args:
        executable: Path to CUDA executable
        chapter_dir: Path to chapter directory
        output_dir: Directory to save nsys-rep file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated nsys-rep file, or None if failed
    """
    if not check_nsys_available():
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on executable name
    exec_name = executable.stem
    nsys_output = output_dir / f"{exec_name}_{variant}.nsys-rep"
    
    # Build nsys command
    nsys_command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "-o",
        str(nsys_output.with_suffix("")),  # nsys adds .nsys-rep automatically
        "-t", "cuda,nvtx,osrt,cublas",
        "-s", "cpu",
        "--cudabacktrace=true",
        "--stats=true",
        str(executable)
    ]
    
    try:
        # nsys profiling timeout: 120 seconds (matches benchmark_harness.nsys_timeout_seconds)
        # nsys needs time to initialize, run executable, and collect profiling data
        result = subprocess.run(
            nsys_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=120,  # Increased from 15s - nsys profiling needs more time
            check=False
        )
        
        if result.returncode == 0 and nsys_output.exists():
            return nsys_output
        else:
            return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def profile_python_benchmark_ncu(
    benchmark: Any,  # Benchmark instance
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> Optional[Path]:
    """Profile a Python benchmark using ncu (NVIDIA Compute Profiler).
    
    Args:
        benchmark: Benchmark instance (already loaded)
        benchmark_path: Path to Python benchmark file (for naming)
        chapter_dir: Path to chapter directory
        output_dir: Directory to save ncu-rep file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated ncu-rep file, or None if failed
    """
    if not check_ncu_available():
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on benchmark name
    benchmark_name = benchmark_path.stem
    ncu_output = output_dir / f"{benchmark_name}_{variant}.ncu-rep"
    
    # Create a temporary wrapper script that runs the benchmark
    wrapper_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    
    try:
        wrapper_script.write(f"""
import sys
from pathlib import Path

# Add chapter directory to path
sys.path.insert(0, r'{chapter_dir}')

# Import and load benchmark
from {benchmark_path.stem} import get_benchmark

benchmark = get_benchmark()
benchmark.setup()

# Warmup
for _ in range(5):
    benchmark.benchmark_fn()

# Profile execution
import torch
if torch.cuda.is_available():
    torch.cuda.synchronize()

benchmark.benchmark_fn()

if torch.cuda.is_available():
    torch.cuda.synchronize()

benchmark.teardown()
""")
        wrapper_script.close()
        
        # Build ncu command
        ncu_command = [
            "ncu",
            "--set", "full",
            "--metrics", "gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active",
            "--replay-mode", "kernel",
            "-o", str(ncu_output.with_suffix("")),  # ncu adds .ncu-rep automatically
            sys.executable,
            wrapper_script.name
        ]
        
        # ncu profiling timeout: 180 seconds (matches benchmark_harness.ncu_timeout_seconds)
        # ncu is slower than nsys and needs more time for metric collection
        result = subprocess.run(
            ncu_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=180,  # Increased from 60s - ncu profiling needs more time
            check=False
        )
        
        # Clean up wrapper script
        try:
            Path(wrapper_script.name).unlink()
        except Exception:
            pass
        
        # Check if file exists (ncu may create file even with non-zero exit code)
        if ncu_output.exists():
            return ncu_output
        # Try alternative path
        alt_path = output_dir / f"{benchmark_name}_{variant}.ncu-rep"
        if alt_path.exists():
            return alt_path
        # Check for any .ncu-rep file matching the pattern
        for ncu_file in output_dir.glob(f"{benchmark_name}_{variant}*.ncu-rep"):
            return ncu_file
        return None
    except Exception:
        # Clean up wrapper script on error
        try:
            Path(wrapper_script.name).unlink()
        except Exception:
            pass
        return None


def profile_cuda_executable_ncu(
    executable: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> Optional[Path]:
    """Profile a CUDA executable using ncu (NVIDIA Compute Profiler).
    
    Args:
        executable: Path to CUDA executable
        chapter_dir: Path to chapter directory
        output_dir: Directory to save ncu-rep file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated ncu-rep file, or None if failed
    """
    if not check_ncu_available():
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on executable name
    exec_name = executable.stem
    ncu_output = output_dir / f"{exec_name}_{variant}.ncu-rep"
    
    # Build ncu command
    ncu_command = [
        "ncu",
        "--set", "full",
        "--metrics", "gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active",
        "--replay-mode", "kernel",
        "-o", str(ncu_output.with_suffix("")),  # ncu adds .ncu-rep automatically
        str(executable)
    ]
    
    try:
        # ncu profiling timeout: 180 seconds (matches benchmark_harness.ncu_timeout_seconds)
        # ncu is slower than nsys and needs more time for metric collection
        result = subprocess.run(
            ncu_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=180,  # Increased from 60s - ncu profiling needs more time
            check=False
        )
        
        # Check if file exists (ncu may create file even with non-zero exit code)
        if ncu_output.exists():
            return ncu_output
        # Try alternative path
        alt_path = output_dir / f"{exec_name}_{variant}.ncu-rep"
        if alt_path.exists():
            return alt_path
        # Check for any .ncu-rep file matching the pattern
        for ncu_file in output_dir.glob(f"{exec_name}_{variant}*.ncu-rep"):
            return ncu_file
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def profile_python_benchmark_torch(
    benchmark: Any,  # Benchmark instance
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> Optional[Path]:
    """Profile a Python benchmark using PyTorch profiler.
    
    Args:
        benchmark: Benchmark instance (already loaded)
        benchmark_path: Path to Python benchmark file (for naming)
        chapter_dir: Path to chapter directory
        output_dir: Directory to save torch trace file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated torch trace JSON file, or None if failed
    """
    if not TORCH_PROFILER_AVAILABLE:
        return None
    
    try:
        import torch.profiler
    except ImportError:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on benchmark name
    benchmark_name = benchmark_path.stem
    torch_output = output_dir / f"{benchmark_name}_{variant}_torch_trace.json"
    
    try:
        # Warmup
        for _ in range(5):
            benchmark.benchmark_fn()
        
        # Profile execution with PyTorch profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            benchmark.benchmark_fn()
            prof.step()
        
        # Export Chrome trace
        prof.export_chrome_trace(str(torch_output))
        
        if torch_output.exists():
            return torch_output
        return None
    except Exception:
        return None


def ensure_cuda_executables_built(chapter_dir: Path) -> bool:
    """Try to build CUDA executables if Makefile exists.
    
    Uses auto-detection to build for the correct GPU architecture (sm_121, sm_103, or sm_100).
    The Makefile will auto-detect the architecture unless ARCH is explicitly set.
    
    Args:
        chapter_dir: Path to chapter directory
        
    Returns:
        True if build succeeded or no Makefile exists, False if build failed
    """
    makefile = chapter_dir / "Makefile"
    if not makefile.exists():
        return True  # No Makefile, assume executables are pre-built or don't exist
    
    env = os.environ.copy()
    make_desc = "default ARCH"
    if detect_supported_arch is not None:
        try:
            arch = detect_supported_arch()
            if arch:
                env["ARCH"] = arch
                make_desc = f"ARCH={arch}"
        except Exception as exc:
            logger.warning(f"  WARNING: Unable to auto-detect CUDA arch for {chapter_dir.name}: {exc}")
    
    try:
        logger.info(f"  Building CUDA executables ({make_desc})...")
        # Explicitly set ARCH so Makefiles consistently target the active GPU
        result = subprocess.run(
            ["make", "-B", "-C", str(chapter_dir), "all"],
            capture_output=True,
            timeout=120,  # Increased timeout - compilation can take time for complex kernels
            check=False,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            logger.warning(f"  WARNING: Make build failed (exit code {result.returncode})")
            if result.stderr:
                logger.warning(f"  Build stderr: {result.stderr[:500]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # Make timed out - compilation takes too long
        logger.warning(f"  WARNING: Make build timed out after 120s - compilation may be too slow or hanging")
        return False
    except Exception as e:
        logger.warning(f"  WARNING: Make build exception: {e}")
        return False


def _test_chapter_impl(chapter_dir: Path, enable_profiling: bool = False, smoke_test: bool = False, timeout_multiplier: float = 1.0, reproducible: bool = False, cold_start: bool = False, iterations: Optional[int] = None, warmup: Optional[int] = None, only_examples: Optional[List[str]] = None) -> Dict[str, Any]:
    """Test all benchmarks in a chapter and return results.
    
    Args:
        chapter_dir: Path to chapter directory
        enable_profiling: If True, generate profiling files (nsys, ncu, PyTorch) alongside benchmarks
        smoke_test: If True, reduce iterations and warmup for smoke testing (CI/quick validation)
        timeout_multiplier: Multiply all timeouts by this factor (e.g., 2.0 = double all timeouts)
        reproducible: If True, set all seeds to 42 and enable deterministic algorithms
        cold_start: If True, perform additional GPU state cleanup (gc.collect()) between benchmarks for cold start measurements. CUDA state is always reset by default.
        iterations: Number of benchmark iterations (overrides smoke_test defaults if provided)
        warmup: Number of warmup iterations (overrides smoke_test defaults if provided)
        only_examples: List of example names to run (e.g., ['moe', 'cutlass']). If None, runs all examples.
    """
    dump_environment_and_capabilities()

    chapter_name = chapter_dir.name
    
    # Set up profiling output directory if profiling is enabled
    profiling_output_dir = None
    if enable_profiling:
        repo_root = chapter_dir.parent
        profiling_output_dir = repo_root / "benchmark_profiles" / chapter_name
        profiling_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which profilers are available
        nsys_avail = check_nsys_available()
        ncu_avail = check_ncu_available()
        # Use module-level check to avoid local variable shadowing issue
        torch_avail = TORCH_PROFILER_AVAILABLE
        
        profilers = []
        if nsys_avail:
            profilers.append("nsys")
        if ncu_avail:
            profilers.append("ncu")
        if torch_avail:
            profilers.append("PyTorch")
        
        if profilers:
            logger.info(f"  Profiling enabled: {', '.join(profilers)} profiling files will be saved to {profiling_output_dir}")
        else:
            logger.warning(f"  Profiling requested but no profilers available - skipping profiling")
            enable_profiling = False
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {chapter_name.upper()}")
    logger.info(f"{'='*80}")
    
    if not torch.cuda.is_available():
        return {
            'chapter': chapter_name,
            'status': 'skipped',
            'reason': 'CUDA not available',
            'benchmarks': [],
            'summary': {
                'total_benchmarks': 0,
                'successful': 0,
                'failed': 0,
                'total_speedup': 0.0,
                'average_speedup': 0.0,
            }
        }
    
    # Reset CUDA state at start of chapter (always, to prevent cascading failures)
    reset_cuda_state()
    # Additional cleanup for cold start mode (includes gc.collect() for more thorough cleanup)
    if cold_start:
        reset_gpu_state()
    
    # Ensure PyTorch inductor cache directory exists to prevent C++ compilation errors
    # (This is also done in env_defaults.py, but we ensure it here as well for safety)
    # Use absolute path to avoid working directory issues
    inductor_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", ".torch_inductor")
    if inductor_cache_dir:
        # Convert relative paths to absolute paths
        if not os.path.isabs(inductor_cache_dir):
            inductor_cache_dir = str(Path.cwd() / inductor_cache_dir)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
        inductor_cache_path = Path(inductor_cache_dir)
        try:
            inductor_cache_path.mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "od").mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "tk").mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # If we can't create the directory, that's okay - env_defaults.py should have handled it
            pass
    
    # Discover Python benchmarks
    logger.info(f"  Discovering Python benchmarks...")
    python_pairs = discover_benchmarks(chapter_dir)
    example_filters = None
    if only_examples:
        example_filters = {name.strip() for name in only_examples if name.strip()}
        if example_filters:
            python_pairs = [
                pair for pair in python_pairs if pair[2] in example_filters
            ]
            logger.info(f"  Filtered to {len(python_pairs)} example(s): {', '.join(sorted(example_filters))}")
    logger.info(f"  Found {len(python_pairs)} Python benchmark pair(s)")
    
    # Discover CUDA benchmarks and ensure executables are built
    logger.info(f"  Discovering CUDA benchmarks...")
    cuda_pairs = discover_cuda_benchmarks(chapter_dir)
    if example_filters:
        cuda_pairs = [
            pair for pair in cuda_pairs if pair[2] in example_filters
        ]
    if cuda_pairs:
        logger.info(f"  Found {len(cuda_pairs)} CUDA benchmark pair(s), ensuring executables are built...")
        ensure_cuda_executables_built(chapter_dir)
    
    if not python_pairs and not cuda_pairs:
        return {
            'chapter': chapter_name,
            'status': 'no_benchmarks',
            'reason': 'No baseline/optimized pairs found',
            'benchmarks': [],
            'summary': {
                'total_benchmarks': 0,
                'successful': 0,
                'failed': 0,
            }
        }
    
    # Create harness for Python benchmarks with explicit timeout to prevent hangs
    # Adjust iterations/warmup based on smoke_test or use provided values
    if iterations is None:
        iterations = 5 if smoke_test else 20
    if warmup is None:
        warmup = 1 if smoke_test else 5
    
    config = BenchmarkConfig(
        iterations=iterations,
        warmup=warmup,
        measurement_timeout_seconds=60,  # Allow heavier smoke workloads before timing out
        timeout_multiplier=timeout_multiplier,  # Apply timeout multiplier from CLI
        enable_memory_tracking=True,  # Enable memory metrics display
        enable_profiling=enable_profiling,  # Respect profiling flag (opt-in via CLI)
        enable_nsys=enable_profiling,  # nsys profiling (gracefully degrades if unavailable)
        enable_ncu=enable_profiling,  # ncu profiling (gracefully degrades if unavailable)
        seed=42 if reproducible else None,  # Set seed for reproducibility
        deterministic=reproducible,  # Enable deterministic algorithms for reproducibility
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    benchmark_results = []
    successful = 0
    failed = 0
    skipped_hw = 0
    skipped_distributed = 0
    informational_skipped = 0
    speedups = []
    informational_examples = INFORMATIONAL_BENCHMARKS.get(chapter_name, set())
    
    # Check GPU count for distributed benchmark detection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Process Python benchmarks
    for baseline_path, optimized_paths, example_name in python_pairs:
        logger.info(f"\n  Example: {example_name}")
        logger.info(f"    Baseline: {baseline_path.name}")
        
        if example_name in informational_examples:
            informational_skipped += 1
            logger.info("    ℹ️ Informational systems demo - documented for reference, not benchmarked.")
            continue
        
        result_entry = {
            'example': example_name,
            'baseline_file': baseline_path.name,
            'baseline_time_ms': None,
             'baseline_throughput': None,
            'optimizations': [],
            'best_speedup': 1.0,
            'status': 'failed',
            'error': None,
        }
        
        # Check if this is a distributed benchmark and we have only 1 GPU
        is_distributed = is_distributed_benchmark(baseline_path)
        if is_distributed and num_gpus == 1:
            skip_reason = f"SKIPPED: Distributed benchmark requires multiple GPUs (found {num_gpus} GPU)"
            logger.warning(f"    WARNING: {skip_reason}")
            result_entry['status'] = 'skipped'
            result_entry['error'] = skip_reason
            result_entry['skip_reason'] = skip_reason
            benchmark_results.append(result_entry)
            skipped_distributed += 1  # Count as skipped, not successful
            continue
        
        # Reset CUDA state before each benchmark pair (always, to prevent cascading failures)
        reset_cuda_state()
        # Additional cleanup for cold start mode (includes gc.collect() for more thorough cleanup)
        if cold_start:
            reset_gpu_state()
        
        # Load and run baseline
        baseline_benchmark = load_benchmark(baseline_path)
        if baseline_benchmark is None:
            result_entry['error'] = 'Failed to load baseline'
            benchmark_results.append(result_entry)
            failed += 1
            reset_cuda_state()  # Reset after failure
            if cold_start:
                reset_gpu_state()
            continue
        
        try:
            # Use benchmark_with_manifest for reproducibility
            run_id = f"{chapter_name}_{example_name}_baseline"
            baseline_run = harness.benchmark_with_manifest(baseline_benchmark, run_id=run_id)
            baseline_result = baseline_run.result
            baseline_timing = baseline_result.timing
            baseline_memory = baseline_result.memory
            baseline_custom_metrics = getattr(baseline_result, "custom_metrics", None) or {}
            if not baseline_custom_metrics:
                getter = getattr(baseline_benchmark, "get_custom_metrics", None)
                if callable(getter):
                    try:
                        metrics = getter()
                        if isinstance(metrics, dict):
                            baseline_custom_metrics = metrics
                    except Exception:
                        baseline_custom_metrics = {}
            baseline_time = baseline_timing.mean_ms if baseline_timing else 0.0
            result_entry['baseline_time_ms'] = baseline_time
            if baseline_custom_metrics:
                result_entry['baseline_custom_metrics'] = baseline_custom_metrics
            
            # Enhanced baseline metrics display with emojis and formatting
            logger.info(f"    Baseline: {format_time_ms(baseline_time)} ms")
            if baseline_timing:
                logger.info(f"      📊 Timing Stats: median={format_time_ms(baseline_timing.median_ms)}ms, "
                      f"min={format_time_ms(baseline_timing.min_ms)}ms, max={format_time_ms(baseline_timing.max_ms)}ms, "
                      f"std={format_time_ms(baseline_timing.std_ms)}ms")
            if baseline_memory and baseline_memory.peak_mb:
                mem_str = f"      💾 Memory: peak={baseline_memory.peak_mb:.2f}MB"
                if baseline_memory.allocated_mb:
                    mem_str += f", allocated={baseline_memory.allocated_mb:.2f}MB"
                logger.info(mem_str)
            if baseline_timing and baseline_timing.percentiles:
                p99 = baseline_timing.percentiles.get(99.0, 0)
                p75 = baseline_timing.percentiles.get(75.0, 0)
                p50 = baseline_timing.percentiles.get(50.0, baseline_timing.median_ms if baseline_timing else 0)
                logger.info(f"      📈 Percentiles: p99={format_time_ms(p99)}ms, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
            baseline_throughput = baseline_result.throughput
            throughput_summary = format_throughput_summary(baseline_throughput)
            if throughput_summary:
                logger.info(f"      ⚡ Throughput: {throughput_summary}")
            serialized_throughput = serialize_throughput(baseline_throughput)
            if serialized_throughput:
                result_entry['baseline_throughput'] = serialized_throughput
            if "scenario_total_phase_ms" in baseline_custom_metrics:
                logger.info(
                    f"      📐 Scenario phase sum: "
                    f"{baseline_custom_metrics['scenario_total_phase_ms']:.3f} ms"
                )
            
            # Profile baseline if profiling is enabled (nsys, ncu, PyTorch)
            if enable_profiling and profiling_output_dir:
                logger.info(f"    Profiling baseline...")
                profiler_results = []
                baseline_metrics = {}
                
                # nsys profiling
                if check_nsys_available():
                    logger.info(f"      nsys...")
                    nsys_path = profile_python_benchmark(
                        baseline_benchmark, baseline_path, chapter_dir, profiling_output_dir, variant="baseline"
                    )
                    if nsys_path:
                        result_entry['baseline_nsys_rep'] = str(nsys_path.relative_to(chapter_dir.parent))
                        profiler_results.append("nsys✓")
                        # Extract metrics
                        nsys_metrics = extract_from_nsys_report(nsys_path)
                        if nsys_metrics:
                            baseline_metrics['nsys'] = nsys_metrics
                    else:
                        profiler_results.append("nsys✗")
                else:
                    profiler_results.append("nsys-")
                
                # ncu profiling
                if check_ncu_available():
                    logger.info(f"ncu...")
                    ncu_path = profile_python_benchmark_ncu(
                        baseline_benchmark, baseline_path, chapter_dir, profiling_output_dir, variant="baseline"
                    )
                    if ncu_path:
                        result_entry['baseline_ncu_rep'] = str(ncu_path.relative_to(chapter_dir.parent))
                        profiler_results.append("ncu✓")
                        # Extract metrics
                        ncu_metrics = extract_from_ncu_report(ncu_path)
                        if ncu_metrics:
                            baseline_metrics['ncu'] = ncu_metrics
                    else:
                        profiler_results.append("ncu✗")
                else:
                    profiler_results.append("ncu-")
                
                # PyTorch profiler
                if TORCH_PROFILER_AVAILABLE:
                    logger.info(f"PyTorch...")
                    torch_path = profile_python_benchmark_torch(
                        baseline_benchmark, baseline_path, chapter_dir, profiling_output_dir, variant="baseline"
                    )
                    if torch_path:
                        result_entry['baseline_torch_trace'] = str(torch_path.relative_to(chapter_dir.parent))
                        profiler_results.append("torch✓")
                        # Extract metrics
                        torch_metrics = extract_from_pytorch_trace(torch_path)
                        if torch_metrics:
                            baseline_metrics['torch'] = torch_metrics
                    else:
                        profiler_results.append("torch✗")
                else:
                    profiler_results.append("torch-")
                
                logger.info(f" ({', '.join(profiler_results)})")
                
                # Display extracted metrics
                if baseline_metrics:
                    logger.info(f"      📈 Profiler Metrics:")
                    if 'nsys' in baseline_metrics:
                        for key, value in baseline_metrics['nsys'].items():
                            logger.info(f"        nsys.{key}: {value:.2f}")
                    if 'ncu' in baseline_metrics:
                        for key, value in baseline_metrics['ncu'].items():
                            logger.info(f"        ncu.{key}: {value:.2f}")
                    if 'torch' in baseline_metrics:
                        for key, value in baseline_metrics['torch'].items():
                            logger.info(f"        torch.{key}: {value:.2f}")
                    result_entry['baseline_profiler_metrics'] = baseline_metrics
        except Exception as e:
            error_str = str(e)
            skip_reason = check_hardware_limitation(error_str)
            
            if skip_reason:
                result_entry['status'] = 'skipped'
                result_entry['error'] = f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}'
                result_entry['skip_reason'] = skip_reason
                logger.warning(f"    WARNING: SKIPPED: {skip_reason}")
                skipped_hw += 1
            else:
                result_entry['error'] = f'Baseline execution failed: {error_str}'
                failed += 1
                maybe_reset_gpu_for_error(error_str, f"{chapter_name}:{example_name}:baseline")
            
            benchmark_results.append(result_entry)
            reset_cuda_state()  # Reset after failure
            # Additional cleanup for cold start mode
            if cold_start:
                reset_gpu_state()
            continue
        
        # Test each optimization
        for optimized_path in optimized_paths:
            opt_name = optimized_path.name
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.py', '')
            if technique == opt_name.replace('optimized_', '').replace('.py', ''):
                technique = 'default'
            
            # Check if optimized benchmark is distributed and we have only 1 GPU
            is_opt_distributed = is_distributed_benchmark(optimized_path)
            if is_opt_distributed and num_gpus == 1:
                skip_reason = f"SKIPPED: Distributed benchmark requires multiple GPUs (found {num_gpus} GPU)"
                logger.warning(f"    WARNING: {opt_name}: {skip_reason}")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'skipped',
                    'error': skip_reason,
                })
                continue
            
            optimized_benchmark = load_benchmark(optimized_path)
            if optimized_benchmark is None:
                logger.error(f"    Testing: {opt_name}... FAILED (load)")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'failed',
                    'error': 'Failed to load',
                })
                continue
            
            try:
                # Reset CUDA state before each optimized benchmark (always, to prevent cascading failures)
                reset_cuda_state()
                # Additional cleanup for cold start mode (includes gc.collect() for more thorough cleanup)
                if cold_start:
                    reset_gpu_state()
                
                # Use benchmark_with_manifest for reproducibility
                opt_run_id = f"{chapter_name}_{example_name}_optimized_{technique}"
                optimized_run = harness.benchmark_with_manifest(optimized_benchmark, run_id=opt_run_id)
                optimized_result = optimized_run.result
                optimized_timing = optimized_result.timing
                optimized_memory = optimized_result.memory
                optimized_custom_metrics = getattr(optimized_result, "custom_metrics", None) or {}
                if not optimized_custom_metrics:
                    getter = getattr(optimized_benchmark, "get_custom_metrics", None)
                    if callable(getter):
                        try:
                            metrics = getter()
                            if isinstance(metrics, dict):
                                optimized_custom_metrics = metrics
                        except Exception:
                            optimized_custom_metrics = {}
                optimized_time = optimized_timing.mean_ms if optimized_timing else 0.0
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                scenario_speedup = None
                b_phase = (result_entry.get('baseline_custom_metrics') or {}).get("scenario_total_phase_ms")
                o_phase = optimized_custom_metrics.get("scenario_total_phase_ms")
                if b_phase and o_phase and o_phase > 0:
                    scenario_speedup = b_phase / o_phase
                    if scenario_speedup > speedup:
                        speedup = scenario_speedup
                
                # Enhanced metrics display with emojis and formatting
                emoji = "🚀" if speedup > 1.0 else "⚠️" if speedup < 1.0 else "="
                logger.info(f"    Testing: {opt_name}... {format_time_ms(optimized_time)} ms ({speedup:.2f}x) {emoji}")
                
                if optimized_timing:
                    logger.info(f"        📊 Timing: median={format_time_ms(optimized_timing.median_ms)}ms, "
                          f"min={format_time_ms(optimized_timing.min_ms)}ms, max={format_time_ms(optimized_timing.max_ms)}ms, "
                          f"std={format_time_ms(optimized_timing.std_ms)}ms")
                
                if optimized_memory and optimized_memory.peak_mb:
                    mem_change = ""
                    if baseline_memory and baseline_memory.peak_mb:
                        diff_mb = optimized_memory.peak_mb - baseline_memory.peak_mb
                        pct_change = (diff_mb / baseline_memory.peak_mb) * 100 if baseline_memory.peak_mb > 0 else 0
                        sign = "+" if diff_mb >= 0 else ""
                        mem_change = f" ({sign}{diff_mb:.2f}MB, {sign}{pct_change:.1f}%)"
                    
                    mem_str = f"        💾 Memory: peak={optimized_memory.peak_mb:.2f}MB{mem_change}"
                    logger.info(mem_str)
                    if optimized_memory.allocated_mb:
                        logger.info(f"                 allocated={optimized_memory.allocated_mb:.2f}MB")
                
                optimized_throughput = optimized_result.throughput
                throughput_summary = format_throughput_summary(optimized_throughput)
                throughput_payload = serialize_throughput(optimized_throughput)
                if throughput_summary:
                    logger.info(f"        ⚡ Throughput: {throughput_summary}")
                
                if "scenario_total_phase_ms" in optimized_custom_metrics:
                    logger.info(
                        f"        📐 Scenario phase sum: "
                        f"{optimized_custom_metrics['scenario_total_phase_ms']:.3f} ms"
                    )
                if scenario_speedup is not None:
                    logger.info(f"        📊 Scenario phase-sum speedup: {scenario_speedup:.2f}x")
                
                if optimized_timing and optimized_timing.percentiles:
                    p99 = optimized_timing.percentiles.get(99.0, 0)
                    p75 = optimized_timing.percentiles.get(75.0, 0)
                    p50 = optimized_timing.percentiles.get(50.0, optimized_timing.median_ms if optimized_timing else 0)
                    p99_speedup = ""
                    if baseline_timing and baseline_timing.percentiles and 99.0 in baseline_timing.percentiles:
                        p99_baseline = baseline_timing.percentiles[99.0]
                        if p99_baseline > 0:
                            p99_speedup = f" ({p99_baseline/p99:.2f}x)" if p99 > 0 else ""
                    logger.info(f"        📈 Percentiles: p99={format_time_ms(p99)}ms{p99_speedup}, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
                
                # Visual speedup bar (always show for consistency)
                bar_length = 40
                if speedup > 1.0:
                    # Improvement: fill bar proportionally to speedup
                    filled = min(int((speedup - 1.0) / max(speedup, 10.0) * bar_length), bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    logger.info(f"        [{bar}] {speedup:.2f}x speedup")
                elif speedup < 1.0:
                    # Regression: show how much slower (distance from 1.0)
                    regress_ratio = (1.0 - speedup)  # e.g., 0.93x = 0.07 (7% slower)
                    # Normalize: 0.5x (50% slower) = full bar, scale linearly
                    filled = min(int(regress_ratio / 0.5 * bar_length), bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    logger.info(f"        [{bar}] {speedup:.2f}x slowdown")
                else:
                    # No change
                    bar = "░" * bar_length
                    logger.info(f"        [{bar}] {speedup:.2f}x (no change)")
                
                opt_result = {
                    'file': opt_name,
                    'technique': technique,
                    'status': 'success',
                    'time_ms': optimized_time,
                    'speedup': speedup,
                }
                if optimized_custom_metrics:
                    opt_result['custom_metrics'] = optimized_custom_metrics
                if scenario_speedup is not None:
                    opt_result['scenario_speedup'] = scenario_speedup
                if throughput_payload:
                    opt_result['throughput'] = throughput_payload
                
                # Profile optimized if profiling is enabled (nsys, ncu, PyTorch)
                if enable_profiling and profiling_output_dir:
                    logger.info(f"\n    Profiling optimized...")
                    profiler_results = []
                    optimized_metrics = {}
                    
                    # nsys profiling
                    if check_nsys_available():
                        logger.info(f"      nsys...")
                        nsys_path = profile_python_benchmark(
                            optimized_benchmark, optimized_path, chapter_dir, profiling_output_dir, 
                            variant=f"optimized_{technique}"
                        )
                        if nsys_path:
                            opt_result['optimized_nsys_rep'] = str(nsys_path.relative_to(chapter_dir.parent))
                            profiler_results.append("nsys✓")
                            # Extract metrics
                            nsys_metrics = extract_from_nsys_report(nsys_path)
                            if nsys_metrics:
                                optimized_metrics['nsys'] = nsys_metrics
                        else:
                            profiler_results.append("nsys✗")
                    else:
                        profiler_results.append("nsys-")
                    
                    # ncu profiling
                    if check_ncu_available():
                        logger.info(f"ncu...")
                        ncu_path = profile_python_benchmark_ncu(
                            optimized_benchmark, optimized_path, chapter_dir, profiling_output_dir,
                            variant=f"optimized_{technique}"
                        )
                        if ncu_path:
                            opt_result['optimized_ncu_rep'] = str(ncu_path.relative_to(chapter_dir.parent))
                            profiler_results.append("ncu✓")
                            # Extract metrics
                            ncu_metrics = extract_from_ncu_report(ncu_path)
                            if ncu_metrics:
                                optimized_metrics['ncu'] = ncu_metrics
                        else:
                            profiler_results.append("ncu✗")
                    else:
                        profiler_results.append("ncu-")
                    
                    # PyTorch profiler
                    if TORCH_PROFILER_AVAILABLE:
                        logger.info(f"PyTorch...")
                        torch_path = profile_python_benchmark_torch(
                            optimized_benchmark, optimized_path, chapter_dir, profiling_output_dir,
                            variant=f"optimized_{technique}"
                        )
                        if torch_path:
                            opt_result['optimized_torch_trace'] = str(torch_path.relative_to(chapter_dir.parent))
                            profiler_results.append("torch✓")
                            # Extract metrics
                            torch_metrics = extract_from_pytorch_trace(torch_path)
                            if torch_metrics:
                                optimized_metrics['torch'] = torch_metrics
                        else:
                            profiler_results.append("torch✗")
                    else:
                        profiler_results.append("torch-")
                    
                    logger.info(f" ({', '.join(profiler_results)})")
                    
                    # Display extracted metrics
                    if optimized_metrics:
                        logger.info(f"        📈 Profiler Metrics:")
                        if 'nsys' in optimized_metrics:
                            for key, value in optimized_metrics['nsys'].items():
                                logger.info(f"          nsys.{key}: {value:.2f}")
                        if 'ncu' in optimized_metrics:
                            for key, value in optimized_metrics['ncu'].items():
                                logger.info(f"          ncu.{key}: {value:.2f}")
                        if 'torch' in optimized_metrics:
                            for key, value in optimized_metrics['torch'].items():
                                logger.info(f"          torch.{key}: {value:.2f}")
                        opt_result['optimized_profiler_metrics'] = optimized_metrics
                
                result_entry['optimizations'].append(opt_result)
                
                if speedup > result_entry['best_speedup']:
                    result_entry['best_speedup'] = speedup
                    speedups.append(speedup)
                
            except Exception as e:
                # Get comprehensive error information with timeout protection
                def safe_get_error_str(exc, timeout_sec=1):
                    """Safely get error string with timeout to prevent hangs."""
                    error_parts = {"type": type(exc).__name__, "str": None, "repr": None}
                    
                    def get_str():
                        try:
                            error_parts["str"] = str(exc)
                        except Exception:
                            pass
                    
                    def get_repr():
                        try:
                            error_parts["repr"] = repr(exc)
                        except Exception:
                            pass
                    
                    # Try to get string representation with timeout
                    import threading
                    t1 = threading.Thread(target=get_str, daemon=True)
                    t2 = threading.Thread(target=get_repr, daemon=True)
                    t1.start()
                    t2.start()
                    t1.join(timeout=timeout_sec)
                    t2.join(timeout=timeout_sec)
                    
                    # Use best available representation
                    if error_parts["str"]:
                        return error_parts["str"]
                    elif error_parts["repr"]:
                        return error_parts["repr"]
                    else:
                        return error_parts["type"]
                
                error_str = safe_get_error_str(e)
                error_full = f"{type(e).__name__}: {error_str}" if error_str else type(e).__name__
                
                # If error string is suspiciously short or empty, try to get more info
                if not error_str or len(error_str.strip()) < 3:
                    import traceback
                    try:
                        tb_lines = traceback.format_exception_only(type(e), e)
                        if tb_lines:
                            error_full = tb_lines[-1].strip()
                            error_str = error_full
                    except Exception:
                        # If even traceback fails, use minimal info
                        error_full = f"{type(e).__name__}: (error message unavailable)"
                
                skip_reason = check_hardware_limitation(error_full)
                
                if skip_reason:
                    logger.warning(f"    Testing: {opt_name}... WARNING: SKIPPED: {skip_reason}")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'skipped',
                        'error': f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}',
                        'skip_reason': skip_reason,
                    })
                    skipped_hw += 1
                else:
                    # Format error message: show full error but truncate if extremely long
                    if len(error_full) > 200:
                        # Try to truncate at word boundary for very long errors
                        truncated = error_full[:197]
                        last_space = truncated.rfind(' ')
                        if last_space > 150:
                            truncated = truncated[:last_space]
                        truncated += "..."
                        logger.error(f"    Testing: {opt_name}... FAILED ({truncated})")
                        logger.error(f"        Full error: {error_full}")
                    else:
                        logger.error(f"    Testing: {opt_name}... FAILED ({error_full})")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'failed',
                        'error': error_full,  # Store full error with type
                    })
                    maybe_reset_gpu_for_error(error_full, f"{chapter_name}:{example_name}:{opt_name}")
                
                reset_cuda_state()  # Reset after failure
                # Additional cleanup for cold start mode
                if cold_start:
                    reset_gpu_state()
        
        required_speedup = PROOF_OF_BENEFIT_REQUIREMENTS.get(chapter_name, {}).get(example_name)
        if (
            required_speedup is not None
            and result_entry.get('baseline_time_ms') is not None
        ):
            result_entry['required_speedup'] = required_speedup
            if result_entry['best_speedup'] < required_speedup:
                msg = (
                    f"Proof-of-benefit requirement not met "
                    f"({result_entry['best_speedup']:.2f}x < {required_speedup:.2f}x)"
                )
                logger.error(f"    ❌ {msg}")
                result_entry['status'] = 'failed'
                result_entry['error'] = msg
                failed += 1
                benchmark_results.append(result_entry)
                reset_cuda_state()
                if cold_start:
                    reset_gpu_state()
                continue

        if result_entry['status'] == 'skipped':
            # Already handled - distributed benchmark skipped
            pass
        elif result_entry.get('baseline_time_ms') is not None:
            # Baseline ran successfully
            if result_entry['optimizations'] and any(opt['status'] == 'success' for opt in result_entry['optimizations']):
                # Has successful optimizations
                result_entry['status'] = 'success'
                successful += 1
            elif result_entry['optimizations'] and all(opt['status'] == 'skipped' for opt in result_entry['optimizations']):
                # All optimizations skipped (e.g., distributed), but baseline ran successfully
                result_entry['status'] = 'success'
                successful += 1
            elif not result_entry['optimizations']:
                # No optimizations to test, but baseline ran successfully
                result_entry['status'] = 'success'
                successful += 1
            else:
                # Baseline ran but optimizations failed
                failed += 1
        else:
            # Baseline failed or didn't run
            failed += 1
        
        benchmark_results.append(result_entry)
        
        # Reset CUDA state after each benchmark pair (always, to ensure clean state)
        reset_cuda_state()
        # Additional cleanup for cold start mode (includes gc.collect() for more thorough cleanup)
        if cold_start:
            reset_gpu_state()
    
    # Process CUDA benchmarks
    for baseline_cu_path, optimized_cu_paths, example_name in cuda_pairs:
        logger.info(f"\n  Example (CUDA): {example_name}")
        
        if example_name in informational_examples:
            informational_skipped += 1
            logger.info("    ℹ️ Informational systems demo - documented for reference, not benchmarked.")
            continue
        
        result_entry = {
            'example': example_name,
            'baseline_file': baseline_cu_path.name,
            'type': 'cuda',
            'baseline_time_ms': None,
            'baseline_throughput': None,
            'optimizations': [],
            'best_speedup': 1.0,
            'status': 'failed',
            'error': None,
        }
        
        # Find baseline executable
        baseline_executable = find_cuda_executable(baseline_cu_path, chapter_dir)
        if baseline_executable is None:
            result_entry['error'] = f'Baseline executable not found for {baseline_cu_path.name}'
            benchmark_results.append(result_entry)
            failed += 1
            reset_cuda_state()  # Reset after failure
            if cold_start:
                reset_gpu_state()
            continue
        
        # Benchmark baseline with explicit timeout
        # Smoke test mode still runs multiple iterations (5) to reduce timing jitter while staying quick
        # Note: Some CUDA benchmarks (like memory transfer) can take 3-5 seconds per run
        # So we need longer timeouts for these benchmarks
        cuda_iterations = 5 if smoke_test else 3
        if smoke_test and chapter_name in SMOKE_CUDA_ITERATION_OVERRIDES:
            cuda_iterations = max(cuda_iterations, SMOKE_CUDA_ITERATION_OVERRIDES[chapter_name])
        cuda_warmup = 0
        cuda_timeout = 20 if smoke_test else 30  # Longer timeout for CUDA benchmarks that do heavy I/O
        logger.info(
            f"    Running baseline executable {baseline_executable.name} "
            f"(runs={cuda_iterations}, timeout={cuda_timeout}s per run)"
        )
        baseline_result = benchmark_cuda_executable(baseline_executable, iterations=cuda_iterations, warmup=cuda_warmup, timeout=cuda_timeout)
        if baseline_result is None:
            result_entry['error'] = f'Baseline execution failed or timed out ({cuda_timeout}s timeout)'
            benchmark_results.append(result_entry)
            failed += 1
            reset_cuda_state()  # Reset after failure
            if cold_start:
                reset_gpu_state()
            continue
        if baseline_result.skip_reason:
            reason = baseline_result.skip_reason
            logger.warning(f"    Baseline executable {baseline_executable.name} SKIPPED: {reason}")
            result_entry['status'] = 'skipped'
            result_entry['error'] = reason
            result_entry['skip_reason'] = reason
            benchmark_results.append(result_entry)
            skipped_hw += 1
            reset_cuda_state()
            if cold_start:
                reset_gpu_state()
            continue
        
        baseline_time = baseline_result.mean_ms
        result_entry['baseline_time_ms'] = baseline_time
        
        # Enhanced baseline metrics display with emojis and formatting (same as Python)
        logger.info(f"    Baseline: {format_time_ms(baseline_time)} ms")
        logger.info(f"      📊 Timing Stats: median={format_time_ms(baseline_result.median_ms)}ms, "
              f"min={format_time_ms(baseline_result.min_ms)}ms, max={format_time_ms(baseline_result.max_ms)}ms, "
              f"std={format_time_ms(baseline_result.std_ms)}ms")
        if baseline_result.percentiles:
            p99 = baseline_result.percentiles.get(99.0, 0)
            p75 = baseline_result.percentiles.get(75.0, 0)
            p50 = baseline_result.percentiles.get(50.0, baseline_result.median_ms)
            logger.info(f"      📈 Percentiles: p99={format_time_ms(p99)}ms, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
        
        # Profile baseline if profiling is enabled (nsys, ncu)
        if enable_profiling and profiling_output_dir:
            logger.info(f"    Profiling baseline...")
            profiler_results = []
            baseline_metrics = {}
            
            # nsys profiling
            if check_nsys_available():
                logger.info(f"      nsys...")
                nsys_path = profile_cuda_executable(
                    baseline_executable, chapter_dir, profiling_output_dir, variant="baseline"
                )
                if nsys_path:
                    result_entry['baseline_nsys_rep'] = str(nsys_path.relative_to(chapter_dir.parent))
                    profiler_results.append("nsys✓")
                    # Extract metrics
                    nsys_metrics = extract_from_nsys_report(nsys_path)
                    if nsys_metrics:
                        baseline_metrics['nsys'] = nsys_metrics
                else:
                    profiler_results.append("nsys✗")
            else:
                profiler_results.append("nsys-")
            
            # ncu profiling
            if check_ncu_available():
                logger.info(f"ncu...")
                ncu_path = profile_cuda_executable_ncu(
                    baseline_executable, chapter_dir, profiling_output_dir, variant="baseline"
                )
                if ncu_path:
                    result_entry['baseline_ncu_rep'] = str(ncu_path.relative_to(chapter_dir.parent))
                    profiler_results.append("ncu✓")
                    # Extract metrics
                    ncu_metrics = extract_from_ncu_report(ncu_path)
                    if ncu_metrics:
                        baseline_metrics['ncu'] = ncu_metrics
                else:
                    profiler_results.append("ncu✗")
            else:
                profiler_results.append("ncu-")
            
            logger.info(f" ({', '.join(profiler_results)})")
            
            # Display extracted metrics
            if baseline_metrics:
                logger.info(f"      📈 Profiler Metrics:")
                if 'nsys' in baseline_metrics:
                    for key, value in baseline_metrics['nsys'].items():
                        logger.info(f"        nsys.{key}: {value:.2f}")
                if 'ncu' in baseline_metrics:
                    for key, value in baseline_metrics['ncu'].items():
                        logger.info(f"        ncu.{key}: {value:.2f}")
                result_entry['baseline_profiler_metrics'] = baseline_metrics
        
        # Test each optimization
        for optimized_cu_path in optimized_cu_paths:
            opt_name = optimized_cu_path.name
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.cu', '')
            if technique == opt_name.replace('optimized_', '').replace('.cu', ''):
                technique = 'default'

            if num_gpus < 2 and cuda_binary_requires_multi_gpu(optimized_cu_path):
                reason = (
                    f"SKIPPED: {opt_name} requires >=2 GPUs (e.g., NVLink/C2C) but only {num_gpus} GPU present"
                )
                logger.warning(f"    WARNING: {reason}")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'skipped',
                    'skip_reason': reason,
                })
                skipped_hw += 1
                continue
            
            optimized_executable = find_cuda_executable(optimized_cu_path, chapter_dir)
            if optimized_executable is None:
                logger.error(f"    Testing: {opt_name}... FAILED (executable not found)")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'failed',
                    'error': 'Executable not found',
                })
                continue

            logger.info(
                f"    Running {opt_name} "
                f"(runs={cuda_iterations}, timeout={cuda_timeout}s per run)"
            )
            optimized_result = benchmark_cuda_executable(optimized_executable, iterations=cuda_iterations, warmup=cuda_warmup, timeout=cuda_timeout)
            if optimized_result is None:
                logger.error(f"    Testing: {opt_name}... FAILED (execution or timeout)")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'failed',
                    'error': f'Execution failed or timed out ({cuda_timeout}s timeout)',
                })
                continue
            if optimized_result.skip_reason:
                reason = optimized_result.skip_reason
                logger.warning(f"    Testing: {opt_name}... WARNING: SKIPPED: {reason}")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'skipped',
                    'skip_reason': reason,
                    'error': f'HARDWARE/SOFTWARE LIMITATION: {reason}',
                })
                skipped_hw += 1
                continue
            
            optimized_time = optimized_result.mean_ms
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            
            # Enhanced metrics display with emojis and formatting (same as Python)
            emoji = "🚀" if speedup > 1.0 else "⚠️" if speedup < 1.0 else "="
            logger.info(f"    Testing: {opt_name}... {format_time_ms(optimized_time)} ms ({speedup:.2f}x) {emoji}")
            
            logger.info(f"        📊 Timing: median={format_time_ms(optimized_result.median_ms)}ms, "
                  f"min={format_time_ms(optimized_result.min_ms)}ms, max={format_time_ms(optimized_result.max_ms)}ms, "
                  f"std={format_time_ms(optimized_result.std_ms)}ms")
            
            if optimized_result.percentiles:
                p99 = optimized_result.percentiles.get(99.0, 0)
                p75 = optimized_result.percentiles.get(75.0, 0)
                p50 = optimized_result.percentiles.get(50.0, optimized_result.median_ms)
                p99_speedup = ""
                if baseline_result.percentiles and 99.0 in baseline_result.percentiles:
                    p99_baseline = baseline_result.percentiles[99.0]
                    if p99_baseline > 0:
                        p99_speedup = f" ({p99_baseline/p99:.2f}x)" if p99 > 0 else ""
                logger.info(f"        📈 Percentiles: p99={format_time_ms(p99)}ms{p99_speedup}, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
            
            # Visual speedup bar (always show for consistency, same as Python)
            bar_length = 40
            if speedup > 1.0:
                # Improvement: fill bar proportionally to speedup
                filled = min(int((speedup - 1.0) / max(speedup, 10.0) * bar_length), bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                logger.info(f"        [{bar}] {speedup:.2f}x speedup")
            elif speedup < 1.0:
                # Regression: show how much slower (distance from 1.0)
                regress_ratio = (1.0 - speedup)  # e.g., 0.93x = 0.07 (7% slower)
                # Normalize: 0.5x (50% slower) = full bar, scale linearly
                filled = min(int(regress_ratio / 0.5 * bar_length), bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                logger.info(f"        [{bar}] {speedup:.2f}x slowdown")
            else:
                # No change
                bar = "░" * bar_length
                logger.info(f"        [{bar}] {speedup:.2f}x (no change)")
            
            opt_result = {
                'file': opt_name,
                'technique': technique,
                'status': 'success',
                'time_ms': optimized_time,
                'speedup': speedup,
            }
            
            # Profile optimized if profiling is enabled (nsys, ncu)
            if enable_profiling and profiling_output_dir:
                logger.info(f"\n    Profiling optimized...")
                profiler_results = []
                optimized_metrics = {}
                
                # nsys profiling
                if check_nsys_available():
                    logger.info(f"      nsys...")
                    nsys_path = profile_cuda_executable(
                        optimized_executable, chapter_dir, profiling_output_dir,
                        variant=f"optimized_{technique}"
                    )
                    if nsys_path:
                        opt_result['optimized_nsys_rep'] = str(nsys_path.relative_to(chapter_dir.parent))
                        profiler_results.append("nsys✓")
                        # Extract metrics
                        nsys_metrics = extract_from_nsys_report(nsys_path)
                        if nsys_metrics:
                            optimized_metrics['nsys'] = nsys_metrics
                    else:
                        profiler_results.append("nsys✗")
                else:
                    profiler_results.append("nsys-")
                
                # ncu profiling
                if check_ncu_available():
                    logger.info(f"ncu...")
                    ncu_path = profile_cuda_executable_ncu(
                        optimized_executable, chapter_dir, profiling_output_dir,
                        variant=f"optimized_{technique}"
                    )
                    if ncu_path:
                        opt_result['optimized_ncu_rep'] = str(ncu_path.relative_to(chapter_dir.parent))
                        profiler_results.append("ncu✓")
                        # Extract metrics
                        ncu_metrics = extract_from_ncu_report(ncu_path)
                        if ncu_metrics:
                            optimized_metrics['ncu'] = ncu_metrics
                    else:
                        profiler_results.append("ncu✗")
                else:
                    profiler_results.append("ncu-")
                
                logger.info(f" ({', '.join(profiler_results)})")
                
                # Display extracted metrics
                if optimized_metrics:
                    logger.info(f"        📈 Profiler Metrics:")
                    if 'nsys' in optimized_metrics:
                        for key, value in optimized_metrics['nsys'].items():
                            logger.info(f"          nsys.{key}: {value:.2f}")
                    if 'ncu' in optimized_metrics:
                        for key, value in optimized_metrics['ncu'].items():
                            logger.info(f"          ncu.{key}: {value:.2f}")
                    opt_result['optimized_profiler_metrics'] = optimized_metrics
            
            result_entry['optimizations'].append(opt_result)
            
            if speedup > result_entry['best_speedup']:
                result_entry['best_speedup'] = speedup
                speedups.append(speedup)
        
        # Determine final status
        if result_entry['optimizations'] and any(opt['status'] == 'success' for opt in result_entry['optimizations']):
            result_entry['status'] = 'success'
            successful += 1
        elif result_entry['optimizations'] and all(opt['status'] == 'skipped' for opt in result_entry['optimizations']):
            result_entry['status'] = 'success'
            successful += 1
        elif not result_entry['optimizations']:
            result_entry['status'] = 'success'
            successful += 1
        else:
            failed += 1
        
        benchmark_results.append(result_entry)
    
    # Calculate summary statistics
    avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
    max_speedup = max(speedups) if speedups else 1.0
    min_speedup = min(speedups) if speedups else 1.0

    logger.info("\n" + "-" * 80)
    logger.info(f"{chapter_name.upper()} SUMMARY")
    total_skipped = skipped_hw + skipped_distributed
    logger.info(
        f"Benchmarks: {len(benchmark_results)} | Successful: {successful} | Failed: {failed} | "
        f"Skipped: {total_skipped} (HW: {skipped_hw}, Dist: {skipped_distributed}) | "
        f"Informational: {informational_skipped}"
    )
    if speedups:
        logger.info(f"Speedups collected: {len(speedups)} | Avg: {avg_speedup:.2f}x | Best: {max_speedup:.2f}x | Worst: {min_speedup:.2f}x")
    else:
        logger.info("No successful optimizations exceeded baseline performance")
    logger.info("-" * 80)
    
    return {
        'chapter': chapter_name,
        'status': 'completed',
        'benchmarks': benchmark_results,
        'summary': {
            'total_benchmarks': len(benchmark_results),
            'successful': successful,
            'failed': failed,
            'skipped_hardware': skipped_hw,
            'skipped_distributed': skipped_distributed,
            'total_skipped': total_skipped,
            'total_speedups': len(speedups),
            'average_speedup': avg_speedup,
            'max_speedup': max_speedup,
            'min_speedup': min_speedup,
            'informational': informational_skipped,
        }
    }


def test_chapter(chapter_dir: Path, enable_profiling: bool = False, smoke_test: bool = False, timeout_multiplier: float = 1.0, reproducible: bool = False, cold_start: bool = False, iterations: Optional[int] = None, warmup: Optional[int] = None, only_examples: Optional[List[str]] = None) -> Dict[str, Any]:
    """Wrapper to toggle smoke-test hint for downstream benchmarks."""
    previous_flag = os.environ.get("BENCHMARK_SMOKE_TEST")
    if smoke_test:
        os.environ["BENCHMARK_SMOKE_TEST"] = "1"
    else:
        os.environ.pop("BENCHMARK_SMOKE_TEST", None)
    try:
        return _test_chapter_impl(
            chapter_dir,
            enable_profiling=enable_profiling,
            smoke_test=smoke_test,
            timeout_multiplier=timeout_multiplier,
            reproducible=reproducible,
            cold_start=cold_start,
            iterations=iterations,
            warmup=warmup,
            only_examples=only_examples,
        )
    finally:
        if previous_flag is None:
            os.environ.pop("BENCHMARK_SMOKE_TEST", None)
        else:
            os.environ["BENCHMARK_SMOKE_TEST"] = previous_flag


def generate_markdown_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate markdown summary report."""
    with open(output_path, 'w') as f:
        f.write("# Benchmark Test Results Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        total_chapters = len(results)
        completed = sum(1 for r in results if r['status'] == 'completed')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        no_benchmarks = sum(1 for r in results if r['status'] == 'no_benchmarks')
        
        total_benchmarks = sum(r['summary']['total_benchmarks'] for r in results)
        total_successful = sum(r['summary']['successful'] for r in results)
        total_failed = sum(r['summary']['failed'] for r in results)
        
        all_speedups = []
        for r in results:
            if r['status'] == 'completed':
                for bench in r['benchmarks']:
                    if bench['status'] == 'success':
                        all_speedups.append(bench['best_speedup'])
        
        avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 1.0
        
        f.write("## Overall Summary\n\n")
        f.write(f"- **Chapters tested:** {completed}/{total_chapters}\n")
        f.write(f"- **Chapters skipped:** {skipped} (CUDA unavailable)\n")
        f.write(f"- **Chapters with no benchmarks:** {no_benchmarks}\n")
        total_skipped_hw = sum(r['summary'].get('skipped_hardware', 0) for r in results)
        total_informational = sum(r['summary'].get('informational', 0) for r in results)
        
        f.write(f"- **Total benchmarks:** {total_benchmarks}\n")
        f.write(f"- **Successful:** {total_successful}\n")
        f.write(f"- **Failed:** {total_failed}\n")
        f.write(f"- **Informational (not benchmarked):** {total_informational}\n")
        if total_skipped_hw > 0:
            f.write(f"- **WARNING: Skipped (hardware/software limitations):** {total_skipped_hw}\n")
        if all_speedups:
            f.write(f"- **Average speedup:** {avg_speedup:.2f}x\n")
            f.write(f"- **Best speedup:** {max(all_speedups):.2f}x\n")
            f.write(f"- **Worst speedup:** {min(all_speedups):.2f}x\n")
        f.write("\n")
        
        # Per-chapter summary table
        f.write("## Per-Chapter Summary\n\n")
        f.write("| Chapter | Status | Benchmarks | Successful | Failed | Avg Speedup | Max Speedup |\n")
        f.write("|---------|--------|------------|------------|--------|-------------|-------------|\n")
        
        for r in sorted(results, key=lambda x: x['chapter']):
            status_emoji = {
                'completed': 'PASS',
                'skipped': 'SKIP',
                'no_benchmarks': 'WARN',
            }.get(r['status'], 'UNKNOWN')
            
            summary = r['summary']
            avg_sp = summary.get('average_speedup', 0.0)
            max_sp = summary.get('max_speedup', 0.0)
            
            f.write(f"| {r['chapter']} | {status_emoji} | {summary['total_benchmarks']} | "
                   f"{summary['successful']} | {summary['failed']} | "
                   f"{avg_sp:.2f}x | {max_sp:.2f}x |\n")
        
        f.write("\n")
        
        # Detailed results per chapter
        f.write("## Detailed Results\n\n")
        for r in sorted(results, key=lambda x: x['chapter']):
            if r['status'] != 'completed':
                continue
            
            f.write(f"### {r['chapter'].upper()}\n\n")
            
            for bench in r['benchmarks']:
                bench_type = bench.get('type', 'python')
                f.write(f"**{bench['example']}**")
                if bench_type == 'cuda':
                    f.write(" *(CUDA)*")
                f.write("\n")
                f.write(f"- Baseline: `{bench['baseline_file']}`")
                if bench['baseline_time_ms']:
                    f.write(f" ({bench['baseline_time_ms']:.2f} ms)")
                profiler_links = []
                if bench.get('baseline_nsys_rep'):
                    profiler_links.append(f"[nsys](./{bench['baseline_nsys_rep']})")
                if bench.get('baseline_ncu_rep'):
                    profiler_links.append(f"[ncu](./{bench['baseline_ncu_rep']})")
                if bench.get('baseline_torch_trace'):
                    profiler_links.append(f"[torch](./{bench['baseline_torch_trace']})")
                if profiler_links:
                    f.write(f" | {' | '.join(profiler_links)}")
                f.write("\n")
                
                if bench['status'] == 'failed':
                    f.write(f"- Failed: {bench.get('error', 'Unknown error')}\n")
                elif bench['status'] == 'skipped':
                    f.write(f"- WARNING: **SKIPPED**: {bench.get('skip_reason', bench.get('error', 'Hardware/software limitation'))}\n")
                else:
                    for opt in bench['optimizations']:
                        if opt['status'] == 'success':
                            f.write(f"- `{opt['file']}`: {opt['time_ms']:.2f} ms ({opt['speedup']:.2f}x speedup)")
                            profiler_links = []
                            if opt.get('optimized_nsys_rep'):
                                profiler_links.append(f"[nsys](./{opt['optimized_nsys_rep']})")
                            if opt.get('optimized_ncu_rep'):
                                profiler_links.append(f"[ncu](./{opt['optimized_ncu_rep']})")
                            if opt.get('optimized_torch_trace'):
                                profiler_links.append(f"[torch](./{opt['optimized_torch_trace']})")
                            if profiler_links:
                                f.write(f" | {' | '.join(profiler_links)}")
                            f.write("\n")
                        elif opt['status'] == 'skipped':
                            f.write(f"- `{opt['file']}`: WARNING: **SKIPPED** - {opt.get('skip_reason', opt.get('error', 'Hardware/software limitation'))}\n")
                        else:
                            f.write(f"- `{opt['file']}`: {opt.get('error', 'Failed')}\n")
                    
                    if bench['best_speedup'] > 1.0:
                        f.write(f"- Best speedup: {bench['best_speedup']:.2f}x\n")
                
                f.write("\n")
            
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test all benchmarks and generate summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--chapter',
        type=str,
        help='Chapter to test (e.g., ch1) or "all" (default: all)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=repo_root / 'benchmark_test_results.json',
        help='Output file path (default: benchmark_test_results.json)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable profiling (generates nsys .nsys-rep, ncu .ncu-rep, and PyTorch trace files for each benchmark)'
    )
    parser.add_argument(
        '--only-examples',
        type=str,
        default=None,
        help='Comma-separated list of example names to run (e.g., "moe,cutlass"). If not specified, runs all examples.'
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run in smoke-test mode (fewer iterations, smaller timeouts) for quicker validation.'
    )
    parser.add_argument(
        '--timeout-multiplier',
        type=float,
        default=1.0,
        help='Multiply all benchmark timeouts by this factor (e.g., 2.0 doubles every timeout).'
    )
    
    args = parser.parse_args()
    
    # Parse --only-examples into a list
    only_examples = None
    if args.only_examples:
        only_examples = [name.strip() for name in args.only_examples.split(",") if name.strip()]
    
    logger.info("=" * 80)
    logger.info("TESTING ALL BENCHMARKS")
    logger.info("=" * 80)
    logger.info("")

    dump_environment_and_capabilities()
    logger.info("")
    
    # Dump hardware capabilities at start
    logger.info("Dumping hardware capabilities...")
    try:
        dump_caps_path = repo_root / "tools" / "utilities" / "dump_hardware_capabilities.py"
        if dump_caps_path.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(dump_caps_path)],
                capture_output=True,
                text=True,
                timeout=15
            )
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        else:
            logger.warning("WARNING: Hardware capabilities script not found")
    except Exception as e:
        logger.warning(f"WARNING: Could not dump hardware capabilities: {e}")
        logger.info("")
    
    # Pre-compile CUDA extensions before running benchmarks
    logger.info("Pre-compiling CUDA extensions...")
    try:
        precompile_path = repo_root / "tools" / "utilities" / "precompile_cuda_extensions.py"
        if precompile_path.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(precompile_path)],
                capture_output=True,
                text=True,
                timeout=60  # 60s - pre-compilation can take time for multiple extensions
            )
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
            precompile_success = result.returncode == 0
            if not precompile_success:
                logger.warning("WARNING: Some CUDA extensions failed to pre-compile")
                logger.warning("   Benchmarks using these extensions may fail")
        else:
            logger.warning("WARNING: Pre-compilation script not found - extensions will compile at runtime")
    except Exception as e:
        logger.warning(f"WARNING: Could not pre-compile CUDA extensions: {e}")
        logger.warning("   Extensions will compile at runtime (may cause segfaults)")
    logger.info("")
    
    # Determine chapters to test
    if args.chapter and args.chapter != 'all':
        chapter_dirs = [repo_root / args.chapter]
    else:
        chapter_dirs = sorted([
            d for d in repo_root.iterdir()
            if d.is_dir() and d.name.startswith('ch') and d.name[2:].isdigit()
        ])
    
    # Test all chapters
    all_results = []
    for chapter_dir in chapter_dirs:
        if not chapter_dir.exists():
            continue
        
        result = test_chapter(
            chapter_dir,
            enable_profiling=args.profile,
            smoke_test=args.smoke_test,
            timeout_multiplier=args.timeout_multiplier,
            only_examples=only_examples
        )
        all_results.append(result)
    
    # Save results
    output_json = args.output
    output_md = args.output.with_suffix('.md')
    
    if args.format in ['json', 'both']:
        with open(output_json, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
            }, f, indent=2)
        logger.info(f"\nJSON results saved to: {output_json}")
    
    if args.format in ['markdown', 'both']:
        generate_markdown_report(all_results, output_md)
        logger.info(f"Markdown report saved to: {output_md}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    total_benchmarks = sum(r['summary']['total_benchmarks'] for r in all_results)
    total_successful = sum(r['summary']['successful'] for r in all_results)
    total_failed = sum(r['summary']['failed'] for r in all_results)
    total_skipped_hw = sum(r['summary'].get('skipped_hardware', 0) for r in all_results)
    total_informational = sum(r['summary'].get('informational', 0) for r in all_results)
    
    logger.info(f"Total benchmarks tested: {total_benchmarks}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Informational (not benchmarked): {total_informational}")
    if total_skipped_hw > 0:
        logger.warning(f"WARNING: Skipped (hardware/software limitations): {total_skipped_hw}")
    
    if total_benchmarks > 0:
        success_rate = (total_successful / total_benchmarks) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    if total_skipped_hw > 0:
        logger.warning(f"\nWARNING: HARDWARE/SOFTWARE LIMITATIONS DETECTED:")
        logger.warning(f"   {total_skipped_hw} benchmarks skipped due to known limitations")
        logger.warning(f"   (e.g., Triton SM 12.1 support, device-side assert cascades)")
        logger.warning(f"   See detailed report for specific skip reasons")
    
    # Calculate overall speedup statistics
    all_speedups = []
    for r in all_results:
        if r['status'] == 'completed':
            for bench in r['benchmarks']:
                if bench['status'] == 'success' and bench['best_speedup'] > 1.0:
                    all_speedups.append(bench['best_speedup'])
    
    if all_speedups:
        logger.info(f"\nSpeedup Statistics:")
        logger.info(f"  Average: {sum(all_speedups)/len(all_speedups):.2f}x")
        logger.info(f"  Best: {max(all_speedups):.2f}x")
        logger.info(f"  Worst: {min(all_speedups):.2f}x")
    
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
