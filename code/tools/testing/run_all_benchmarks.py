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

import sys
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any, Optional
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
    
    # Triton SM architecture issues - REMOVED!
    # arch_config.py patches Triton to work with SM 12.1, so Triton benchmarks should work.
    # If they fail, it's a real error, not a hardware limitation - don't skip.
    
    # Device-side assert cascades - these should be prevented by reset_cuda_state()
    # But if they still occur, it's a transient state issue, not a hardware limitation
    if 'device-side assert' in error_lower or 'cudaerrorassert' in error_lower:
        # Don't skip - reset should handle this. Return None to let it fail normally.
        return None
    
    # Torch.compile timeouts - should reduce model size instead of skipping
    # Only skip if it's a true timeout AND we can't reduce model size
    # But for now, let's not skip - the benchmark should handle this
    if 'timeout' in error_lower and 'compile' in error_lower:
        # Don't skip - benchmarks should use safe_compile() with timeout or reduce model size
        return None
    
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


def benchmark_cuda_executable(executable: Path, iterations: int = 20, warmup: int = 5, timeout: int = 15) -> Optional[CudaBenchmarkResult]:
    """Benchmark a CUDA executable and return statistical results.
    
    Args:
        executable: Path to CUDA executable
        iterations: Number of benchmark iterations
        warmup: Number of warmup runs
        timeout: Timeout per run in seconds (default: 15 seconds to prevent hangs)
        
    Returns:
        CudaBenchmarkResult with statistical measures, or None if failed
    """
    times_ms = []
    
    # Warmup runs
    for _ in range(warmup):
        try:
            subprocess.run(
                [str(executable)],
                capture_output=True,
                timeout=timeout,
                check=False
            )
        except subprocess.TimeoutExpired:
            # Timeout occurred - kill any remaining processes
            import os
            import signal
            try:
                # Try to kill any child processes that might still be running
                os.killpg(os.getpgid(0), signal.SIGTERM)
            except:
                pass
            return None
    
    # Benchmark runs
    for _ in range(iterations):
        try:
            start = time.perf_counter()
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                timeout=timeout,
                check=False
            )
            end = time.perf_counter()
            
            if result.returncode == 0:
                elapsed_ms = (end - start) * 1000.0
                times_ms.append(elapsed_ms)
        except subprocess.TimeoutExpired:
            # Timeout occurred - kill any remaining processes
            import os
            import signal
            try:
                # Try to kill any child processes that might still be running
                os.killpg(os.getpgid(0), signal.SIGTERM)
            except:
                pass
            return None
    
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
        
        result = subprocess.run(
            nsys_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=15,
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
        result = subprocess.run(
            nsys_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=15,
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
        
        result = subprocess.run(
            ncu_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=60,  # ncu can be slow
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
        result = subprocess.run(
            ncu_command,
            cwd=str(chapter_dir),
            capture_output=True,
            timeout=60,  # ncu can be slow
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
    
    Args:
        chapter_dir: Path to chapter directory
        
    Returns:
        True if build succeeded or no Makefile exists, False if build failed
    """
    makefile = chapter_dir / "Makefile"
    if not makefile.exists():
        return True  # No Makefile, assume executables are pre-built or don't exist
    
    try:
        result = subprocess.run(
            ["make", "-C", str(chapter_dir)],
            capture_output=True,
            timeout=60,  # 60s - compilation can take time for complex kernels
            check=False
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # Make timed out - compilation takes too long
        print(f"  WARNING: Make build timed out after 60s - compilation may be too slow or hanging")
        return False
    except Exception:
        return False


def test_chapter(chapter_dir: Path, enable_profiling: bool = False) -> Dict[str, Any]:
    """Test all benchmarks in a chapter and return results.
    
    Args:
        chapter_dir: Path to chapter directory
        enable_profiling: If True, generate profiling files (nsys, ncu, PyTorch) alongside benchmarks
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
            print(f"  Profiling enabled: {', '.join(profilers)} profiling files will be saved to {profiling_output_dir}")
        else:
            print(f"  WARNING: Profiling requested but no profilers available - skipping profiling")
            enable_profiling = False
    
    print(f"\n{'='*80}")
    print(f"Testing {chapter_name.upper()}")
    print(f"{'='*80}")
    
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
    
    # Reset CUDA state at start of chapter
    reset_cuda_state()
    
    # Discover Python benchmarks
    print(f"  Discovering Python benchmarks...", flush=True)
    python_pairs = discover_benchmarks(chapter_dir)
    print(f"  Found {len(python_pairs)} Python benchmark pair(s)", flush=True)
    
    # Discover CUDA benchmarks and ensure executables are built
    print(f"  Discovering CUDA benchmarks...", flush=True)
    cuda_pairs = discover_cuda_benchmarks(chapter_dir)
    if cuda_pairs:
        print(f"  Found {len(cuda_pairs)} CUDA benchmark pair(s), ensuring executables are built...")
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
    config = BenchmarkConfig(
        iterations=20,
        warmup=5,
        timeout_seconds=15,  # 15 second timeout per benchmark to prevent hangs
        enable_memory_tracking=True  # Enable memory metrics display
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    benchmark_results = []
    successful = 0
    failed = 0
    skipped_hw = 0
    speedups = []
    
    # Process Python benchmarks
    for baseline_path, optimized_paths, example_name in python_pairs:
        print(f"\n  Example: {example_name}")
        print(f"    Baseline: {baseline_path.name}")
        
        result_entry = {
            'example': example_name,
            'baseline_file': baseline_path.name,
            'baseline_time_ms': None,
            'optimizations': [],
            'best_speedup': 1.0,
            'status': 'failed',
            'error': None,
        }
        
        # Reset CUDA state before each benchmark pair
        reset_cuda_state()
        
        # Load and run baseline
        baseline_benchmark = load_benchmark(baseline_path)
        if baseline_benchmark is None:
            result_entry['error'] = 'Failed to load baseline'
            benchmark_results.append(result_entry)
            failed += 1
            continue
        
        try:
            baseline_result = harness.benchmark(baseline_benchmark)
            baseline_time = baseline_result.mean_ms
            result_entry['baseline_time_ms'] = baseline_time
            
            # Enhanced baseline metrics display with emojis and formatting
            print(f"    Baseline: {format_time_ms(baseline_time)} ms")
            print(f"      📊 Timing Stats: median={format_time_ms(baseline_result.median_ms)}ms, "
                  f"min={format_time_ms(baseline_result.min_ms)}ms, max={format_time_ms(baseline_result.max_ms)}ms, "
                  f"std={format_time_ms(baseline_result.std_ms)}ms")
            if baseline_result.memory_peak_mb:
                mem_str = f"      💾 Memory: peak={baseline_result.memory_peak_mb:.2f}MB"
                if baseline_result.memory_allocated_mb:
                    mem_str += f", allocated={baseline_result.memory_allocated_mb:.2f}MB"
                print(mem_str)
            if baseline_result.percentiles:
                p99 = baseline_result.percentiles.get(99.0, 0)
                p75 = baseline_result.percentiles.get(75.0, 0)
                p50 = baseline_result.percentiles.get(50.0, baseline_result.median_ms)
                print(f"      📈 Percentiles: p99={format_time_ms(p99)}ms, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
            
            # Profile baseline if profiling is enabled (nsys, ncu, PyTorch)
            if enable_profiling and profiling_output_dir:
                print(f"    Profiling baseline...", flush=True)
                profiler_results = []
                baseline_metrics = {}
                
                # nsys profiling
                if check_nsys_available():
                    print(f"      nsys...", end=' ', flush=True)
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
                    print(f"ncu...", end=' ', flush=True)
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
                    print(f"PyTorch...", end=' ', flush=True)
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
                
                print(f" ({', '.join(profiler_results)})")
                
                # Display extracted metrics
                if baseline_metrics:
                    print(f"      📈 Profiler Metrics:")
                    if 'nsys' in baseline_metrics:
                        for key, value in baseline_metrics['nsys'].items():
                            print(f"        nsys.{key}: {value:.2f}")
                    if 'ncu' in baseline_metrics:
                        for key, value in baseline_metrics['ncu'].items():
                            print(f"        ncu.{key}: {value:.2f}")
                    if 'torch' in baseline_metrics:
                        for key, value in baseline_metrics['torch'].items():
                            print(f"        torch.{key}: {value:.2f}")
                    result_entry['baseline_profiler_metrics'] = baseline_metrics
        except Exception as e:
            error_str = str(e)
            skip_reason = check_hardware_limitation(error_str)
            
            if skip_reason:
                result_entry['status'] = 'skipped'
                result_entry['error'] = f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}'
                result_entry['skip_reason'] = skip_reason
                print(f"    WARNING: SKIPPED: {skip_reason}")
                skipped_hw += 1
            else:
                result_entry['error'] = f'Baseline execution failed: {error_str}'
                failed += 1
            
            benchmark_results.append(result_entry)
            reset_cuda_state()  # Reset after failure
            continue
        
        # Test each optimization
        for optimized_path in optimized_paths:
            opt_name = optimized_path.name
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.py', '')
            if technique == opt_name.replace('optimized_', '').replace('.py', ''):
                technique = 'default'
            
            optimized_benchmark = load_benchmark(optimized_path)
            if optimized_benchmark is None:
                print(f"    Testing: {opt_name}... FAILED (load)")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'failed',
                    'error': 'Failed to load',
                })
                continue
            
            try:
                # Reset CUDA state before each optimized benchmark
                reset_cuda_state()
                
                optimized_result = harness.benchmark(optimized_benchmark)
                optimized_time = optimized_result.mean_ms
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                
                # Enhanced metrics display with emojis and formatting
                emoji = "🚀" if speedup > 1.0 else "⚠️" if speedup < 1.0 else "="
                print(f"    Testing: {opt_name}... {format_time_ms(optimized_time)} ms ({speedup:.2f}x) {emoji}")
                
                print(f"        📊 Timing: median={format_time_ms(optimized_result.median_ms)}ms, "
                      f"min={format_time_ms(optimized_result.min_ms)}ms, max={format_time_ms(optimized_result.max_ms)}ms, "
                      f"std={format_time_ms(optimized_result.std_ms)}ms")
                
                if optimized_result.memory_peak_mb:
                    mem_change = ""
                    if baseline_result.memory_peak_mb:
                        diff_mb = optimized_result.memory_peak_mb - baseline_result.memory_peak_mb
                        pct_change = (diff_mb / baseline_result.memory_peak_mb) * 100 if baseline_result.memory_peak_mb > 0 else 0
                        sign = "+" if diff_mb >= 0 else ""
                        mem_change = f" ({sign}{diff_mb:.2f}MB, {sign}{pct_change:.1f}%)"
                    
                    mem_str = f"        💾 Memory: peak={optimized_result.memory_peak_mb:.2f}MB{mem_change}"
                    print(mem_str)
                    if optimized_result.memory_allocated_mb:
                        print(f"                 allocated={optimized_result.memory_allocated_mb:.2f}MB")
                
                if optimized_result.percentiles:
                    p99 = optimized_result.percentiles.get(99.0, 0)
                    p75 = optimized_result.percentiles.get(75.0, 0)
                    p50 = optimized_result.percentiles.get(50.0, optimized_result.median_ms)
                    p99_speedup = ""
                    if baseline_result.percentiles and 99.0 in baseline_result.percentiles:
                        p99_baseline = baseline_result.percentiles[99.0]
                        if p99_baseline > 0:
                            p99_speedup = f" ({p99_baseline/p99:.2f}x)" if p99 > 0 else ""
                    print(f"        📈 Percentiles: p99={format_time_ms(p99)}ms{p99_speedup}, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
                
                # Visual speedup bar (always show for consistency)
                bar_length = 40
                if speedup > 1.0:
                    # Improvement: fill bar proportionally to speedup
                    filled = min(int((speedup - 1.0) / max(speedup, 10.0) * bar_length), bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"        [{bar}] {speedup:.2f}x speedup")
                elif speedup < 1.0:
                    # Regression: show how much slower (distance from 1.0)
                    regress_ratio = (1.0 - speedup)  # e.g., 0.93x = 0.07 (7% slower)
                    # Normalize: 0.5x (50% slower) = full bar, scale linearly
                    filled = min(int(regress_ratio / 0.5 * bar_length), bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"        [{bar}] {speedup:.2f}x slowdown")
                else:
                    # No change
                    bar = "░" * bar_length
                    print(f"        [{bar}] {speedup:.2f}x (no change)")
                
                opt_result = {
                    'file': opt_name,
                    'technique': technique,
                    'status': 'success',
                    'time_ms': optimized_time,
                    'speedup': speedup,
                }
                
                # Profile optimized if profiling is enabled (nsys, ncu, PyTorch)
                if enable_profiling and profiling_output_dir:
                    print(f"\n    Profiling optimized...", flush=True)
                    profiler_results = []
                    optimized_metrics = {}
                    
                    # nsys profiling
                    if check_nsys_available():
                        print(f"      nsys...", end=' ', flush=True)
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
                        print(f"ncu...", end=' ', flush=True)
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
                        print(f"PyTorch...", end=' ', flush=True)
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
                    
                    print(f" ({', '.join(profiler_results)})")
                    
                    # Display extracted metrics
                    if optimized_metrics:
                        print(f"        📈 Profiler Metrics:")
                        if 'nsys' in optimized_metrics:
                            for key, value in optimized_metrics['nsys'].items():
                                print(f"          nsys.{key}: {value:.2f}")
                        if 'ncu' in optimized_metrics:
                            for key, value in optimized_metrics['ncu'].items():
                                print(f"          ncu.{key}: {value:.2f}")
                        if 'torch' in optimized_metrics:
                            for key, value in optimized_metrics['torch'].items():
                                print(f"          torch.{key}: {value:.2f}")
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
                    print(f"    Testing: {opt_name}... WARNING: SKIPPED: {skip_reason}")
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
                        print(f"    Testing: {opt_name}... FAILED ({truncated})")
                        print(f"        Full error: {error_full}")
                    else:
                        print(f"    Testing: {opt_name}... FAILED ({error_full})")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'failed',
                        'error': error_full,  # Store full error with type
                    })
                
                reset_cuda_state()  # Reset after failure
        
        if result_entry['status'] == 'skipped':
            # Already handled
            pass
        elif result_entry['optimizations'] and any(opt['status'] == 'success' for opt in result_entry['optimizations']):
            result_entry['status'] = 'success'
            successful += 1
        else:
            failed += 1
        
        benchmark_results.append(result_entry)
        
        # Reset CUDA state after each benchmark pair
        reset_cuda_state()
    
    # Process CUDA benchmarks
    for baseline_cu_path, optimized_cu_paths, example_name in cuda_pairs:
        print(f"\n  Example (CUDA): {example_name}")
        
        result_entry = {
            'example': example_name,
            'baseline_file': baseline_cu_path.name,
            'type': 'cuda',
            'baseline_time_ms': None,
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
            continue
        
        # Benchmark baseline with explicit timeout
        baseline_result = benchmark_cuda_executable(baseline_executable, iterations=20, warmup=5, timeout=15)
        if baseline_result is None:
            result_entry['error'] = 'Baseline execution failed or timed out (15s timeout)'
            benchmark_results.append(result_entry)
            failed += 1
            continue
        
        baseline_time = baseline_result.mean_ms
        result_entry['baseline_time_ms'] = baseline_time
        
        # Enhanced baseline metrics display with emojis and formatting (same as Python)
        print(f"    Baseline: {format_time_ms(baseline_time)} ms")
        print(f"      📊 Timing Stats: median={format_time_ms(baseline_result.median_ms)}ms, "
              f"min={format_time_ms(baseline_result.min_ms)}ms, max={format_time_ms(baseline_result.max_ms)}ms, "
              f"std={format_time_ms(baseline_result.std_ms)}ms")
        if baseline_result.percentiles:
            p99 = baseline_result.percentiles.get(99.0, 0)
            p75 = baseline_result.percentiles.get(75.0, 0)
            p50 = baseline_result.percentiles.get(50.0, baseline_result.median_ms)
            print(f"      📈 Percentiles: p99={format_time_ms(p99)}ms, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
        
        # Profile baseline if profiling is enabled (nsys, ncu)
        if enable_profiling and profiling_output_dir:
            print(f"    Profiling baseline...", flush=True)
            profiler_results = []
            baseline_metrics = {}
            
            # nsys profiling
            if check_nsys_available():
                print(f"      nsys...", end=' ', flush=True)
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
                print(f"ncu...", end=' ', flush=True)
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
            
            print(f" ({', '.join(profiler_results)})")
            
            # Display extracted metrics
            if baseline_metrics:
                print(f"      📈 Profiler Metrics:")
                if 'nsys' in baseline_metrics:
                    for key, value in baseline_metrics['nsys'].items():
                        print(f"        nsys.{key}: {value:.2f}")
                if 'ncu' in baseline_metrics:
                    for key, value in baseline_metrics['ncu'].items():
                        print(f"        ncu.{key}: {value:.2f}")
                result_entry['baseline_profiler_metrics'] = baseline_metrics
        
        # Test each optimization
        for optimized_cu_path in optimized_cu_paths:
            opt_name = optimized_cu_path.name
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.cu', '')
            if technique == opt_name.replace('optimized_', '').replace('.cu', ''):
                technique = 'default'
            
            optimized_executable = find_cuda_executable(optimized_cu_path, chapter_dir)
            if optimized_executable is None:
                print(f"    Testing: {opt_name}... FAILED (executable not found)")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'failed',
                    'error': 'Executable not found',
                })
                continue
            
            optimized_result = benchmark_cuda_executable(optimized_executable, iterations=20, warmup=5, timeout=15)
            if optimized_result is None:
                print(f"    Testing: {opt_name}... FAILED (execution or timeout)")
                result_entry['optimizations'].append({
                    'file': opt_name,
                    'technique': technique,
                    'status': 'failed',
                    'error': 'Execution failed or timed out (15s timeout)',
                })
                continue
            
            optimized_time = optimized_result.mean_ms
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            
            # Enhanced metrics display with emojis and formatting (same as Python)
            emoji = "🚀" if speedup > 1.0 else "⚠️" if speedup < 1.0 else "="
            print(f"    Testing: {opt_name}... {format_time_ms(optimized_time)} ms ({speedup:.2f}x) {emoji}")
            
            print(f"        📊 Timing: median={format_time_ms(optimized_result.median_ms)}ms, "
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
                print(f"        📈 Percentiles: p99={format_time_ms(p99)}ms{p99_speedup}, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
            
            # Visual speedup bar (always show for consistency, same as Python)
            bar_length = 40
            if speedup > 1.0:
                # Improvement: fill bar proportionally to speedup
                filled = min(int((speedup - 1.0) / max(speedup, 10.0) * bar_length), bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"        [{bar}] {speedup:.2f}x speedup")
            elif speedup < 1.0:
                # Regression: show how much slower (distance from 1.0)
                regress_ratio = (1.0 - speedup)  # e.g., 0.93x = 0.07 (7% slower)
                # Normalize: 0.5x (50% slower) = full bar, scale linearly
                filled = min(int(regress_ratio / 0.5 * bar_length), bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"        [{bar}] {speedup:.2f}x slowdown")
            else:
                # No change
                bar = "░" * bar_length
                print(f"        [{bar}] {speedup:.2f}x (no change)")
            
            opt_result = {
                'file': opt_name,
                'technique': technique,
                'status': 'success',
                'time_ms': optimized_time,
                'speedup': speedup,
            }
            
            # Profile optimized if profiling is enabled (nsys, ncu)
            if enable_profiling and profiling_output_dir:
                print(f"\n    Profiling optimized...", flush=True)
                profiler_results = []
                optimized_metrics = {}
                
                # nsys profiling
                if check_nsys_available():
                    print(f"      nsys...", end=' ', flush=True)
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
                    print(f"ncu...", end=' ', flush=True)
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
                
                print(f" ({', '.join(profiler_results)})")
                
                # Display extracted metrics
                if optimized_metrics:
                    print(f"        📈 Profiler Metrics:")
                    if 'nsys' in optimized_metrics:
                        for key, value in optimized_metrics['nsys'].items():
                            print(f"          nsys.{key}: {value:.2f}")
                    if 'ncu' in optimized_metrics:
                        for key, value in optimized_metrics['ncu'].items():
                            print(f"          ncu.{key}: {value:.2f}")
                    opt_result['optimized_profiler_metrics'] = optimized_metrics
            
            result_entry['optimizations'].append(opt_result)
            
            if speedup > result_entry['best_speedup']:
                result_entry['best_speedup'] = speedup
                speedups.append(speedup)
        
        # Determine final status
        if result_entry['optimizations'] and any(opt['status'] == 'success' for opt in result_entry['optimizations']):
            result_entry['status'] = 'success'
            successful += 1
        else:
            failed += 1
        
        benchmark_results.append(result_entry)
    
    # Calculate summary statistics
    avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
    max_speedup = max(speedups) if speedups else 1.0
    min_speedup = min(speedups) if speedups else 1.0

    print("\n" + "-" * 80)
    print(f"{chapter_name.upper()} SUMMARY")
    print(f"Benchmarks: {len(benchmark_results)} | Successful: {successful} | Failed: {failed} | Skipped: {skipped_hw}")
    if speedups:
        print(f"Speedups collected: {len(speedups)} | Avg: {avg_speedup:.2f}x | Best: {max_speedup:.2f}x | Worst: {min_speedup:.2f}x")
    else:
        print("No successful optimizations exceeded baseline performance")
    print("-" * 80)
    
    return {
        'chapter': chapter_name,
        'status': 'completed',
        'benchmarks': benchmark_results,
        'summary': {
            'total_benchmarks': len(benchmark_results),
            'successful': successful,
            'failed': failed,
            'skipped_hardware': skipped_hw,
            'total_speedups': len(speedups),
            'average_speedup': avg_speedup,
            'max_speedup': max_speedup,
            'min_speedup': min_speedup,
        }
    }


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
        
        f.write(f"- **Total benchmarks:** {total_benchmarks}\n")
        f.write(f"- **Successful:** {total_successful}\n")
        f.write(f"- **Failed:** {total_failed}\n")
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
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TESTING ALL BENCHMARKS")
    print("=" * 80)
    print()

    dump_environment_and_capabilities()
    print()
    
    # Dump hardware capabilities at start
    print("Dumping hardware capabilities...")
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
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        else:
            print("WARNING: Hardware capabilities script not found")
    except Exception as e:
        print(f"WARNING: Could not dump hardware capabilities: {e}")
        print()
    
    # Pre-compile CUDA extensions before running benchmarks
    print("Pre-compiling CUDA extensions...")
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
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            precompile_success = result.returncode == 0
            if not precompile_success:
                print("WARNING: Some CUDA extensions failed to pre-compile")
                print("   Benchmarks using these extensions may fail")
        else:
            print("WARNING: Pre-compilation script not found - extensions will compile at runtime")
    except Exception as e:
        print(f"WARNING: Could not pre-compile CUDA extensions: {e}")
        print("   Extensions will compile at runtime (may cause segfaults)")
    print()
    
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
        
        result = test_chapter(chapter_dir, enable_profiling=args.profile)
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
        print(f"\nJSON results saved to: {output_json}")
    
    if args.format in ['markdown', 'both']:
        generate_markdown_report(all_results, output_md)
        print(f"Markdown report saved to: {output_md}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_benchmarks = sum(r['summary']['total_benchmarks'] for r in all_results)
    total_successful = sum(r['summary']['successful'] for r in all_results)
    total_failed = sum(r['summary']['failed'] for r in all_results)
    total_skipped_hw = sum(r['summary'].get('skipped_hardware', 0) for r in all_results)
    
    print(f"Total benchmarks tested: {total_benchmarks}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    if total_skipped_hw > 0:
        print(f"WARNING: Skipped (hardware/software limitations): {total_skipped_hw}")
    
    if total_benchmarks > 0:
        success_rate = (total_successful / total_benchmarks) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if total_skipped_hw > 0:
        print(f"\nWARNING: HARDWARE/SOFTWARE LIMITATIONS DETECTED:")
        print(f"   {total_skipped_hw} benchmarks skipped due to known limitations")
        print(f"   (e.g., Triton SM 12.1 support, device-side assert cascades)")
        print(f"   See detailed report for specific skip reasons")
    
    # Calculate overall speedup statistics
    all_speedups = []
    for r in all_results:
        if r['status'] == 'completed':
            for bench in r['benchmarks']:
                if bench['status'] == 'success' and bench['best_speedup'] > 1.0:
                    all_speedups.append(bench['best_speedup'])
    
    if all_speedups:
        print(f"\nSpeedup Statistics:")
        print(f"  Average: {sum(all_speedups)/len(all_speedups):.2f}x")
        print(f"  Best: {max(all_speedups):.2f}x")
        print(f"  Worst: {min(all_speedups):.2f}x")
    
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
