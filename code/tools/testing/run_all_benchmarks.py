#!/usr/bin/env python3
"""Run every single benchmark/example and summarize results.

This script:
1. Discovers all baseline/optimized pairs across all chapters
2. Runs actual benchmarks using BenchmarkHarness
3. Collects performance metrics (speedup, latency, throughput, etc.)
4. Generates a comprehensive summary report

Usage:
    python tools/testing/run_all_benchmarks.py [--targets chX chY:example] [--format json|markdown|both]
"""

import sys
import os
from pathlib import Path
import json
import argparse
import shlex
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import statistics
import math
from dataclasses import dataclass, fields
import threading
from contextlib import ExitStack
import copy

# Force NVCC line info so Nsight/torch traces carry file/line metadata
os.environ["NVCCFLAGS"] = f"-lineinfo {os.environ.get('NVCCFLAGS', '')}".strip()

# Ensure repository root on sys.path before importing helpers
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from arch_config import configure_optimizations as _configure_arch_optimizations

_configure_arch_optimizations()

from common.python import compile_utils as _compile_utils_patch  # noqa: F401

from common.python.env_defaults import apply_env_defaults, dump_environment_and_capabilities

apply_env_defaults()

import torch
import subprocess
import time
import os
import tempfile
from common.python.chapter_compare_template import (
    discover_benchmarks,
    load_benchmark,
    get_last_load_error,
)
from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig
from common.python.benchmark_defaults import BenchmarkDefaults, set_defaults
from common.python.smoke import is_smoke_mode, set_smoke_mode
from common.python.run_manifest import reset_gpu_state, get_git_info
from common.python.gpu_telemetry import format_gpu_telemetry, query_gpu_telemetry
try:
    from common.python.cuda_binary_benchmark import detect_supported_arch
except ImportError:  # pragma: no cover - optional dependency during docs builds
    detect_supported_arch = None  # type: ignore[assignment]
from common.python.expectations import (
    ExpectationsStore,
    METRIC_DIRECTIONS,
    detect_expectation_key,
)
from tools.verification.verify_all_benchmarks import chapter_slug, resolve_target_chapters

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
                
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        pass  # Profile file corrupt or unexpected format
    
    return metrics


# Examples that demonstrate techniques but may not show speedup (educational demos, analysis tools)
# These are valuable for showing HOW to implement something, even if not faster for this workload
INFORMATIONAL_BENCHMARKS: Dict[str, Set[str]] = {
    # Ch3: NUMA awareness demo shows the technique (outcome depends on system topology)
    "ch3": {"numa_unaware"},
    # Ch4: DataParallel demo shows basic parallelism pattern (requires multi-GPU)
    "ch4": {"dataparallel_basic"},
    # Ch12: Graph CUDA demos show graph capture patterns
    "ch12": {"graph_cuda"},
    # Ch14: Sliding window bench uses FlexAttention which may not have full Blackwell support
    "ch14": {"sliding_window_bench"},
    # Ch15: Disaggregated inference demos show architecture patterns (multi-GPU)
    "ch15": {"disaggregated_inference", "inference_placement"},
    # Ch16: Paged attention demos show memory management technique (value is memory efficiency)
    "ch16": {"paged_attention", "paged_attention_blackwell", "piece_graphs"},
    # Ch17: Pipeline parallelism and routing demos (multi-GPU)
    "ch17": {"pipeline_parallelism", "prefill_decode_disagg"},
    # Ch18: Speculative decoding demos show technique patterns
    "ch18": {"speculative_decoding_multi_draft", "flexdecoding_graphs"},
    # Ch19: NVFP4 is cutting-edge and may not be faster than BF16 yet
    "ch19": {"nvfp4_training"},
    # Labs: Dynamic router demos show routing patterns
    "dynamic_router": {"dynamic_router", "router_vectorized"},
    # Labs: Persistent decode demos show technique patterns
    "persistent_decode": {"kv_locality_microbench", "persistent_decode_cuda"},
}

# Note: The following have been moved to tools/:
# - ch2/uma_memory_reporting -> tools/diagnostics/uma_memory/
# - ch10/roofline -> tools/analysis/roofline/
# - ch15/moe_validation -> tools/verification/moe/
# - speculative_decode/spec_config_sweep -> tools/analysis/speculative_decode/
# - occupancy_tuning/proton_* -> tools/profiling/occupancy_tuning/

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


EXPECTATION_THROUGHPUT_FIELDS: Tuple[str, ...] = (
    "requests_per_s",
    "tokens_per_s",
    "samples_per_s",
    "goodput",
    "latency_ms",
)


def expectation_example_key(example_name: str, bench_type: str) -> str:
    bench_type = (bench_type or "python").lower()
    if bench_type == "python":
        return example_name
    return f"{example_name}_{bench_type}"


def find_best_optimization_entry(optimizations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best_entry: Optional[Dict[str, Any]] = None
    best_speed = float("-inf")
    for opt in optimizations or []:
        if opt.get("status") != "succeeded":
            continue
        speed = float(opt.get("speedup") or 0.0)
        if speed > best_speed:
            best_entry = opt
            best_speed = speed
    return best_entry


def start_progress_watchdog(
    logger,
    chapter_name: str,
    warn_after: float = 300.0,
    ping_every: float = 90.0,
):
    """Spawn a background watchdog that emits progress heartbeats and hang warnings."""
    state = {
        "last_progress": time.time(),
        "last_note": "initializing chapter",
        "warned": False,
        "active": True,
    }
    stop_event = threading.Event()
    ping_every = max(30.0, ping_every)
    warn_after = max(ping_every * 2.0, warn_after)

    def heartbeat() -> None:
        while not stop_event.wait(ping_every):
            if not state["active"]:
                break
            elapsed = time.time() - state["last_progress"]
            if elapsed >= warn_after:
                if not state["warned"]:
                    logger.warning(
                        "    ⏱️ No benchmark progress in %s for %.0fs (last completed: %s)",
                        chapter_name,
                        elapsed,
                        state["last_note"],
                    )
                    state["warned"] = True
            else:
                logger.info(
                    "    …%s still running (last completed: %s, %.0fs ago)",
                    chapter_name,
                    state["last_note"],
                    elapsed,
                )

    thread = threading.Thread(
        target=heartbeat,
        name=f"{chapter_name}_progress_watchdog",
        daemon=True,
    )
    thread.start()

    def record(note: str) -> None:
        gap = time.time() - state["last_progress"]
        state["last_progress"] = time.time()
        state["last_note"] = note
        if state["warned"]:
            logger.info(
                "    ✅ Progress resumed after %.0fs (now at %s)",
                gap,
                note,
            )
            state["warned"] = False

    def stop() -> None:
        state["active"] = False
        stop_event.set()
        thread.join(timeout=1.0)

    return record, stop


def _capture_metric(metrics: Dict[str, float], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    if key in METRIC_DIRECTIONS:
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            pass


def _capture_payload(metrics: Dict[str, float], prefix: str, payload: Optional[Dict[str, Any]]) -> None:
    if not payload or not isinstance(payload, dict):
        return
    for field in EXPECTATION_THROUGHPUT_FIELDS:
        metric_key = f"{prefix}.{field}"
        if metric_key not in METRIC_DIRECTIONS:
            continue
        value = payload.get(field)
        if isinstance(value, (int, float)):
            metrics[metric_key] = float(value)


def _capture_custom_metrics(metrics: Dict[str, float], prefix: str, payload: Optional[Dict[str, Any]]) -> None:
    if not payload or not isinstance(payload, dict):
        return
    for key, value in payload.items():
        metric_key = f"{prefix}.{key}"
        if metric_key not in METRIC_DIRECTIONS:
            continue
        if isinstance(value, (int, float)):
            metrics[metric_key] = float(value)


def collect_expectation_metrics(result_entry: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    metrics: Dict[str, float] = {}
    _capture_metric(metrics, "best_speedup", result_entry.get("best_speedup"))
    _capture_metric(metrics, "baseline_time_ms", result_entry.get("baseline_time_ms"))
    _capture_metric(metrics, "baseline_p75_ms", result_entry.get("baseline_p75_ms"))
    _capture_metric(metrics, "baseline_p90_ms", result_entry.get("baseline_p90_ms"))

    baseline_throughput = result_entry.get("baseline_throughput")
    _capture_payload(metrics, "baseline_throughput", baseline_throughput)

    baseline_custom = result_entry.get("baseline_custom_metrics")
    _capture_custom_metrics(metrics, "baseline_custom", baseline_custom)

    best_opt = find_best_optimization_entry(result_entry.get("optimizations", []))
    if best_opt:
        _capture_metric(metrics, "best_optimized_time_ms", best_opt.get("time_ms"))
        _capture_metric(metrics, "best_optimized_speedup", best_opt.get("speedup"))
        _capture_metric(metrics, "best_optimized_p75_ms", best_opt.get("p75_ms"))
        _capture_metric(metrics, "best_optimized_p90_ms", best_opt.get("p90_ms"))
        _capture_payload(metrics, "best_optimized_throughput", best_opt.get("throughput"))
        _capture_custom_metrics(metrics, "best_optimized_custom", best_opt.get("custom_metrics"))

    return metrics, best_opt


def build_expectation_metadata(
    result_entry: Dict[str, Any],
    best_opt: Optional[Dict[str, Any]],
    git_commit: Optional[str],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "example": result_entry.get("example"),
        "type": result_entry.get("type", "python"),
    }
    if git_commit:
        metadata["git_commit"] = git_commit
    if best_opt:
        metadata["best_optimization"] = best_opt.get("technique") or best_opt.get("file")
        metadata["best_optimization_file"] = best_opt.get("file")
        metadata["best_optimization_speedup"] = best_opt.get("speedup")
        metadata["best_optimization_time_ms"] = best_opt.get("time_ms")
    return metadata


def _format_metric_value(value: Optional[float]) -> str:
    if not isinstance(value, (int, float)):
        return str(value) if value is not None else "n/a"
    if math.isnan(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.5f}"


def _format_profiler_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return _format_metric_value(float(value))
    return str(value)


def log_profiler_metrics_table(
    logger,
    metrics: Dict[str, Dict[str, Any]],
    indent: str = "",
) -> None:
    rows: List[Dict[str, str]] = []
    for profiler_name, profiler_metrics in sorted(metrics.items()):
        for metric_key, value in sorted(profiler_metrics.items()):
            rows.append(
                {
                    "Profiler": profiler_name,
                    "Metric": metric_key,
                    "Value": _format_profiler_value(value),
                }
            )
    if not rows:
        logger.info(f"{indent}(no profiler metrics)")
        return
    headers = ["Profiler", "Metric", "Value"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row.get(header, "")))
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    divider = "-+-".join("-" * widths[header] for header in headers)
    logger.info(f"{indent}{header_line}")
    logger.info(f"{indent}{divider}")
    for row in rows:
        line = " | ".join(row.get(header, "").ljust(widths[header]) for header in headers)
        logger.info(f"{indent}{line}")


def log_expectation_evaluation(
    logger,
    evaluation: Optional[Any],
    repo_root: Path,
) -> None:
    if evaluation is None:
        return
    rel_path = None
    if evaluation.expectation_path:
        try:
            rel_path = evaluation.expectation_path.relative_to(repo_root)
        except ValueError:
            rel_path = evaluation.expectation_path
    header = f"    Expectations [{evaluation.hardware_key}]"
    if rel_path:
        header += f": {rel_path}"
    logger.info(header)
    comparisons = evaluation.comparisons or []
    if not comparisons:
        logger.info("      (no expectation comparisons)")
        return

    headers = ["Metric", "Observed", "Expected", "Delta", "Δ%", "Status"]
    rows: List[Dict[str, str]] = []
    for comp in comparisons:
        delta_pct = comp.get("delta_pct")
        pct_str = "n/a"
        if delta_pct is not None and not math.isinf(delta_pct):
            pct_str = f"{delta_pct:+.2f}%"
        elif delta_pct is not None and math.isinf(delta_pct):
            pct_str = "+inf%" if delta_pct > 0 else "-inf%"
        rows.append(
            {
                "Metric": comp.get("metric", ""),
                "Observed": _format_metric_value(comp.get("observed")),
                "Expected": _format_metric_value(comp.get("expected")),
                "Delta": _format_metric_value(comp.get("delta")),
                "Δ%": pct_str,
                "Status": comp.get("status", ""),
            }
        )
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row.get(header, "")))
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    divider = "-+-".join("-" * widths[header] for header in headers)
    logger.info(f"      {header_line}")
    logger.info(f"      {divider}")
    for row in rows:
        line = " | ".join(row.get(header, "").ljust(widths[header]) for header in headers)
        logger.info(f"      {line}")


def reset_cuda_state():
    """Reset CUDA state to prevent cascading failures.
    
    Performs thorough cleanup:
    - Garbage collection to release Python references
    - Empty CUDA cache to free GPU memory
    - Synchronize all CUDA streams
    - Reset peak memory stats
    - Clear torch.compile caches
    """
    import gc
    
    # Force garbage collection first to release Python references to CUDA tensors
    gc.collect()
    
    try:
        if torch.cuda.is_available():
            # Synchronize first to complete pending operations
            torch.cuda.synchronize()
            # Empty cache to free all unreferenced memory
            torch.cuda.empty_cache()
            # Reset memory tracking stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            # Another sync to ensure cleanup is complete
            torch.cuda.synchronize()
    except RuntimeError:
        pass  # CUDA not initialized or error
    
    # Clear torch.compile caches to prevent stale compiled code
    try:
        torch._dynamo.reset()
    except Exception:
        pass  # May not be available in all versions
    
    # Second GC pass to clean up any CUDA objects freed above
    gc.collect()


def clean_build_directories(chapter_dir: Path) -> None:
    """Clean build directories to prevent stale lock issues.
    
    Removes stale lock files and cleans torch extensions cache for the chapter.
    This prevents hangs from build locks left by crashed processes.
    """
    import shutil
    
    try:
        from common.python.build_utils import ensure_clean_build_directory
    except ImportError:
        ensure_clean_build_directory = None
    
    # Clean chapter build directory - more aggressive: remove lock files directly
    build_dir = chapter_dir / "build"
    if build_dir.exists():
        # Remove all lock files first
        for lock_file in build_dir.glob("**/*.lock"):
            try:
                lock_file.unlink()
            except Exception:
                pass
        for lock_file in build_dir.glob("**/lock"):
            try:
                lock_file.unlink()
            except Exception:
                pass
        for lock_file in build_dir.glob("**/.ninja_lock"):
            try:
                lock_file.unlink()
            except Exception:
                pass
        # Then run the standard cleanup
        if ensure_clean_build_directory:
            try:
                ensure_clean_build_directory(build_dir, max_lock_age_seconds=30)
            except Exception as e:
                logger.warning(f"Failed to clean build directory {build_dir}: {e}")
    
    # Clean torch extensions cache for this chapter
    torch_ext_dir = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache" / "torch_extensions"))
    chapter_name = chapter_dir.name
    for ext_dir in torch_ext_dir.glob(f"py*/{chapter_name}*"):
        # Remove all lock files
        for lock_file in ext_dir.glob("**/*.lock"):
            try:
                lock_file.unlink()
            except Exception:
                pass
        for lock_file in ext_dir.glob("**/lock"):
            try:
                lock_file.unlink()
            except Exception:
                pass
        if ensure_clean_build_directory:
            try:
                ensure_clean_build_directory(ext_dir, max_lock_age_seconds=30)
            except Exception:
                pass  # Best effort cleanup
    
    # Also clean torch inductor cache locks
    inductor_cache = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", ".torch_inductor"))
    if inductor_cache.exists():
        for lock_file in inductor_cache.glob("**/*.lock"):
            try:
                lock_file.unlink()
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
        
        # Some examples gate their execution behind helper functions such as
        # skip_if_insufficient_gpus() even if they do not import torch.distributed.
        # Treat those as multi-GPU benchmarks so we can skip them up-front.
        multi_gpu_markers = (
            "skip_if_insufficient_gpus",
            "requires_multiple_gpus",
            "requires_multi_gpu",
            "MultiGPUBenchmark",
            "MIN_GPUS_REQUIRED",
        )
        has_explicit_multi_gpu_guard = any(marker in content for marker in multi_gpu_markers)

        # A benchmark is distributed if it has distributed imports AND operations,
        # OR if it explicitly uses NCCL backend, OR if it contains explicit
        # multi-GPU guard helpers.
        return (
            (has_dist_import and has_dist_ops)
            or has_nccl
            or (has_world_size and has_rank and has_dist_ops)
            or has_explicit_multi_gpu_guard
        )
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


def determine_cuda_skip_reason(
    cu_file: Path,
    chapter_dir: Path,
    build_success: bool,
    build_warning: Optional[str],
) -> str:
    """Return a best-effort skip reason when a CUDA executable is unavailable."""
    name = cu_file.stem.lower()
    if not build_success:
        detail = build_warning or "CUDA Makefile build failed"
        return f"SKIPPED: CUDA executables were not built ({detail})"
    
    # Implementation-only translation units are wrapped by *_host.cu or *_static.cu files.
    wrapper_candidates = [
        chapter_dir / f"{cu_file.stem}_host.cu",
        chapter_dir / f"{cu_file.stem}_static.cu",
        chapter_dir / f"{cu_file.stem}_host_sm121",
        chapter_dir / f"{cu_file.stem}_static_sm121",
    ]
    if any(candidate.exists() for candidate in wrapper_candidates):
        return (
            f"SKIPPED: {cu_file.name} is included by a host wrapper and is not built as a standalone binary on this platform"
        )
    
    if "tcgen05" in name:
        return (
            "SKIPPED: tcgen05 kernels require Tensor Memory Accelerator instructions that "
            "are unavailable in this CUDA 13.0 toolchain"
        )
    if "dsmem" in name or "cluster" in name:
        return (
            "SKIPPED: Distributed Shared Memory / thread block clusters are disabled on this GPU/driver"
        )
    if "pipeline" in name and "warp_specialized" in name:
        return (
            "SKIPPED: Warp specialization cluster pipelines require thread block cluster hardware support"
        )
    if "dynamic_parallelism" in name and "host" not in name:
        return (
            "SKIPPED: This dynamic parallelism driver is compiled only via the *_host.cu wrapper"
        )
    
    return (
        "SKIPPED: CUDA executable not available on this architecture (implementation-only or omitted from Makefile)"
    )


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
    percentiles_to_compute = [25.0, 50.0, 75.0, 90.0, 99.0]
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
        Path(wrapper_script.name).unlink(missing_ok=True)
        
        if result.returncode == 0 and nsys_output.exists():
            return nsys_output
        else:
            return None
    except (subprocess.SubprocessError, OSError):
        # Clean up wrapper script on error
        Path(wrapper_script.name).unlink(missing_ok=True)
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
        Path(wrapper_script.name).unlink(missing_ok=True)
        
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
    except (subprocess.SubprocessError, OSError):
        # Clean up wrapper script on error
        Path(wrapper_script.name).unlink(missing_ok=True)
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
            with_modules=True,
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


def ensure_cuda_executables_built(chapter_dir: Path) -> Tuple[bool, Optional[str]]:
    """Try to build CUDA executables if Makefile exists.
    
    Uses auto-detection to build for the correct GPU architecture (sm_121, sm_103, or sm_100).
    The Makefile will auto-detect the architecture unless ARCH is explicitly set.
    
    Args:
        chapter_dir: Path to chapter directory
        
    Returns:
        Tuple of (success flag, optional failure reason)
    """
    makefile = chapter_dir / "Makefile"
    if not makefile.exists():
        return True, None  # No Makefile, assume executables are pre-built or don't exist
    
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
        # Clean build directory before building to prevent stale lock issues
        build_dir = chapter_dir / "build"
        if build_dir.exists():
            try:
                from common.python.build_utils import ensure_clean_build_directory
                ensure_clean_build_directory(build_dir)
            except ImportError:
                pass  # build_utils not available
            except Exception:
                pass  # Ignore cleanup errors
        
        logger.info(f"  Building CUDA executables ({make_desc})...")
        # Explicitly set ARCH so Makefiles consistently target the active GPU
        result = subprocess.run(
            ["make", "-B", "-C", str(chapter_dir), "all"],
            capture_output=True,
            timeout=300,  # Increased timeout for CUDA JIT compilation (can take 60+ seconds)
            check=False,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            logger.warning(f"  WARNING: Make build failed (exit code {result.returncode})")
            if result.stderr:
                logger.warning(f"  Build stderr: {result.stderr[:500]}")
            failure_snippet = (result.stderr or "").strip().splitlines()
            reason = failure_snippet[0] if failure_snippet else f"Make exited with code {result.returncode}"
            return False, reason
        return True, None
    except subprocess.TimeoutExpired:
        # Make timed out - compilation takes too long
        logger.warning(f"  WARNING: Make build timed out after 300s - compilation may be too slow or hanging")
        return False, "Make build timed out after 300s"
    except Exception as e:
        logger.warning(f"  WARNING: Make build exception: {e}")
        return False, str(e)


def _test_chapter_impl(
    chapter_dir: Path,
    enable_profiling: bool = False,
    profile_type: str = "none",
    timeout_multiplier: float = 1.0,
    reproducible: bool = False,
    cold_start: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    only_examples: Optional[List[str]] = None,
    accept_regressions: bool = False,
    ncu_metric_set: str = "auto",
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    env_passthrough: Optional[List[str]] = None,
    target_extra_args: Optional[Dict[str, List[str]]] = None,
    # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
    verify_input: bool = True,
    verify_output: bool = True,
    # LLM analysis and patching options
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
) -> Dict[str, Any]:
    """Test all benchmarks in a chapter and return results.
    
    Args:
        chapter_dir: Path to chapter directory
        enable_profiling: If True, generate profiling files (nsys, ncu, PyTorch) alongside benchmarks
        timeout_multiplier: Multiply all timeouts by this factor (e.g., 2.0 = double all timeouts)
        reproducible: If True, set all seeds to 42 and force deterministic algorithms (slower fallbacks; ops without deterministic support may fail)
        cold_start: If True, perform additional GPU state cleanup (gc.collect()) between benchmarks for cold start measurements. CUDA state is always reset by default.
        iterations: Number of benchmark iterations (defaults to 20 if not provided)
        warmup: Number of warmup iterations (defaults to 5 if not provided)
        only_examples: List of example names to run (e.g., ['moe', 'cutlass']). If None, runs all examples.
        launch_via: Launcher to use ('python' or 'torchrun')
        nproc_per_node: torchrun --nproc_per_node value
        nnodes: torchrun --nnodes value
        rdzv_backend: torchrun rendezvous backend
        rdzv_endpoint: torchrun rendezvous endpoint
        env_passthrough: Environment variables to pass through to subprocess launches
        target_extra_args: Optional per-target arg overrides (target -> list of CLI args)
    """
    logger.info("launch_via arg=%s nproc_per_node=%s nnodes=%s", launch_via, nproc_per_node, nnodes)
    dump_environment_and_capabilities()

    chapter_id = chapter_slug(chapter_dir, repo_root)
    chapter_name = chapter_id.replace("/", "_")

    # Set up profiling output directory if profiling is enabled
    profiling_output_dir = None
    if enable_profiling:
        profiling_output_dir = repo_root / "benchmark_profiles" / Path(chapter_id)
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

    expectation_hardware_key = detect_expectation_key()
    expectations_store = ExpectationsStore(
        chapter_dir,
        expectation_hardware_key,
        accept_regressions=accept_regressions,
    )
    expectations_disabled = is_smoke_mode()
    if expectations_disabled:
        expectations_store.evaluate = lambda *args, **kwargs: None  # type: ignore[assignment]
    try:
        expectation_path = expectations_store.path.relative_to(repo_root)
    except ValueError:
        expectation_path = expectations_store.path
    logger.info(f"  Expectations key: {expectation_hardware_key} (file: {expectation_path})")
    git_commit = None
    try:
        git_commit = get_git_info().get("commit")
    except Exception:
        git_commit = None
    
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
    
    # Clean build directories to prevent stale lock issues (before any GPU operations)
    logger.info(f"  Cleaning build directories...")
    clean_build_directories(chapter_dir)
    
    # Reset CUDA state at start of chapter (always, to prevent cascading failures)
    logger.info(f"  Resetting GPU state...")
    reset_cuda_state()
    reset_gpu_state()  # Full GPU reset between chapters for clean state
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
        logger.info(f"  Requested examples: {only_examples}")
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
    cuda_build_ok = True
    cuda_build_warning = None
    if cuda_pairs:
        logger.info(f"  Found {len(cuda_pairs)} CUDA benchmark pair(s), ensuring executables are built...")
        cuda_build_ok, cuda_build_warning = ensure_cuda_executables_built(chapter_dir)
    
    total_benchmarks = len(python_pairs) + len(cuda_pairs)
    logger.info(f"  Benchmark counts -> python: {len(python_pairs)}, cuda: {len(cuda_pairs)}, total: {total_benchmarks}")
    if not total_benchmarks:
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
    
    progress_recorder = None
    stop_watchdog = None
    if total_benchmarks:
        progress_recorder, stop_watchdog = start_progress_watchdog(
            logger,
            chapter_name,
        )

    # Create harness for Python benchmarks with explicit timeout to prevent hangs
    cli_iterations_provided = iterations is not None
    cli_warmup_provided = warmup is not None

    if iterations is None:
        iterations = 20
    if warmup is None:
        warmup = 5
    
    try:
        from common.python.benchmark_defaults import get_defaults as _get_defaults  # type: ignore
        _defaults_obj = _get_defaults()
    except Exception:
        _defaults_obj = None

    base_config = BenchmarkConfig(
        iterations=iterations,
        warmup=warmup,
        measurement_timeout_seconds=180,  # Increased for CUDA JIT compilation (can take 30+ seconds)
        setup_timeout_seconds=120,  # Allow more time for setup (CUDA extension loading)
        timeout_multiplier=timeout_multiplier,  # Apply timeout multiplier from CLI
        enable_memory_tracking=True,  # Enable memory metrics display
        enable_profiling=enable_profiling,  # Respect profiling flag (opt-in via CLI)
        enable_nsys=enable_profiling,  # nsys profiling (gracefully degrades if unavailable)
        enable_ncu=enable_profiling,  # ncu profiling (gracefully degrades if unavailable)
        seed=42 if reproducible else None,  # Set seed for reproducibility
        deterministic=reproducible,  # Enable deterministic algorithms for reproducibility
        ncu_metric_set=ncu_metric_set,
        profile_type=profile_type if enable_profiling else "none",
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        env_passthrough=env_passthrough or None,
        target_extra_args=target_extra_args or {},
    )
    logger.info("base_config launch_via=%s", base_config.launch_via)
    if profiling_output_dir:
        base_config.profiling_output_dir = str(profiling_output_dir)

    # Fields that should not be overridden by individual benchmarks
    protected_fields: Set[str] = {"enable_memory_tracking"}  # Always track memory for analysis
    if enable_profiling:
        protected_fields.update({"enable_profiling", "enable_nsys", "enable_ncu", "enable_nvtx", "profile_type"})

    def _merged_config(benchmark_obj) -> BenchmarkConfig:
        merged = copy.deepcopy(base_config)
        defaults_obj = _defaults_obj
        bench_config = getattr(benchmark_obj, "get_config", None)
        override = bench_config() if callable(bench_config) else None
        if override:
            for field in fields(BenchmarkConfig):
                value = getattr(override, field.name, None)
                if value is None:
                    continue
                if field.name == "timeout_multiplier":
                    # Preserve CLI-provided timeout scaling so per-benchmark defaults
                    # cannot silently stretch timeouts.
                    continue
                if field.name == "launch_via":
                    base_value = getattr(merged, field.name, None)
                    default_value = getattr(defaults_obj, field.name, None) if defaults_obj else None
                    def _normalize_launch(val):
                        if val is None:
                            return None
                        if hasattr(val, "value"):
                            return str(getattr(val, "value")).lower()
                        return str(val).lower()
                    base_norm = _normalize_launch(base_value)
                    default_norm = _normalize_launch(default_value)
                    value_norm = _normalize_launch(value)
                    # Preserve CLI-provided launcher when the benchmark config only supplies the default
                    if (
                        base_norm is not None
                        and default_norm is not None
                        and base_norm != default_norm
                        and value_norm == default_norm
                    ):
                        continue
                if field.name == "target_extra_args":
                    if value:
                        merged.target_extra_args = {
                            **(getattr(merged, "target_extra_args", {}) or {}),
                            **value,
                        }
                    continue
                if field.name == "iterations" and cli_iterations_provided:
                    continue
                if field.name == "warmup" and cli_warmup_provided:
                    continue
                if field.name == "env_passthrough" and not value:
                    continue
                if field.name in protected_fields and getattr(base_config, field.name):
                    continue
                setattr(merged, field.name, copy.deepcopy(value))
        merged._sync_execution_mode()
        merged._sync_launch_via()
        # Prevent benchmark-specific defaults from widening CLI/base timeouts.
        timeout_fields = (
            "setup_timeout_seconds",
            "warmup_timeout_seconds",
            "measurement_timeout_seconds",
            "profiling_timeout_seconds",
            "nsys_timeout_seconds",
            "ncu_timeout_seconds",
            "proton_timeout_seconds",
            "timeout_seconds",
        )
        for field_name in timeout_fields:
            base_value = getattr(base_config, field_name, None)
            merged_value = getattr(merged, field_name, None)
            if base_value is None or merged_value is None:
                continue
            try:
                if merged_value > base_value:
                    setattr(merged, field_name, base_value)
            except TypeError:
                # Non-numeric timeout values are unexpected; keep merged value.
                pass
        merged.timeout_multiplier = getattr(base_config, "timeout_multiplier", merged.timeout_multiplier)
        return merged

    def _run_with_config(benchmark_obj, run_id: str, target_label: Optional[str] = None):
        merged = _merged_config(benchmark_obj)
        if target_label and getattr(merged, "target_label", None) is None:
            merged.target_label = target_label
        logger.info("merged config launch_via=%s execution_mode=%s", merged.launch_via, merged.execution_mode)
        local_harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=merged)
        return local_harness.benchmark_with_manifest(benchmark_obj, run_id=run_id)
    
    benchmark_results = []
    successful = 0
    failed_error = 0
    failed_regression = 0
    skipped_hw = 0
    skipped_distributed = 0
    informational_skipped = 0
    speedups = []
    informational_examples = INFORMATIONAL_BENCHMARKS.get(chapter_name, set())
    
    # Check GPU count for distributed benchmark detection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    done_count = 0

    def mark_progress(example_label: str) -> None:
        nonlocal done_count
        done_count += 1
        if progress_recorder:
            progress_recorder(f"{chapter_name}:{example_label} ({done_count}/{total_benchmarks})")

    from contextlib import ExitStack

    with ExitStack() as cleanup_stack:
        if stop_watchdog:
            cleanup_stack.callback(stop_watchdog)

        # Process Python benchmarks
        for baseline_path, optimized_paths, example_name in python_pairs:
            logger.info(f"\n  Example: {example_name}")
            logger.info(f"    Baseline: {baseline_path.name}")
        
            if example_name in informational_examples:
                informational_skipped += 1
                logger.info("    ℹ️ Informational systems demo - documented for reference, not benchmarked.")
                mark_progress(example_name)
                continue
        
            result_entry = {
                'example': example_name,
                'type': 'python',
                'baseline_file': baseline_path.name,
                'baseline_time_ms': None,
                'baseline_throughput': None,
                'baseline_memory_mb': None,  # Peak memory for baseline
                'optimizations': [],
                'best_speedup': 1.0,
                'best_memory_savings_pct': 0.0,  # Memory reduction percentage
                'optimization_goal': 'speed',  # Primary goal: speed, memory, throughput
                'status': 'failed_error',
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
                mark_progress(example_name)
                continue
        
            # Reset CUDA state before each benchmark pair (always, to prevent cascading failures)
            reset_cuda_state()
            # Additional cleanup for cold start mode (includes gc.collect() for more thorough cleanup)
            if cold_start:
                reset_gpu_state()
            
            # Load and run baseline
            baseline_benchmark = load_benchmark(baseline_path)
            if baseline_benchmark is None:
                load_error = get_last_load_error() or ""
                skip_reason = check_hardware_limitation(load_error)
                if skip_reason:
                    result_entry['status'] = 'skipped'
                    result_entry['error'] = f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}'
                    result_entry['skip_reason'] = skip_reason
                    benchmark_results.append(result_entry)
                    skipped_hw += 1
                else:
                    result_entry['error'] = 'Failed to load baseline'
                    benchmark_results.append(result_entry)
                    failed_error += 1
                reset_cuda_state()  # Reset after failure or skip
                if cold_start:
                    reset_gpu_state()
                continue
            
            try:
                # Use benchmark_with_manifest for reproducibility
                run_id = f"{chapter_name}_{example_name}_baseline"
                baseline_run = _run_with_config(
                    baseline_benchmark,
                    run_id=run_id,
                    target_label=f"{chapter_name}:{example_name}",
                )
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
                
                # Capture baseline memory
                if baseline_memory and baseline_memory.peak_mb:
                    result_entry['baseline_memory_mb'] = baseline_memory.peak_mb
                
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
                    result_entry['baseline_percentiles'] = dict(baseline_timing.percentiles)
                    if p75 is not None:
                        result_entry['baseline_p75_ms'] = p75
                    p90 = baseline_timing.p90_ms or baseline_timing.percentiles.get(90.0)
                    if p90 is not None:
                        result_entry['baseline_p90_ms'] = p90
                baseline_throughput = baseline_result.throughput
                throughput_summary = format_throughput_summary(baseline_throughput)
                if throughput_summary:
                    logger.info(f"      ⚡ Throughput: {throughput_summary}")
                serialized_throughput = serialize_throughput(baseline_throughput)
                if serialized_throughput:
                    result_entry['baseline_throughput'] = serialized_throughput
                baseline_gpu_metrics = getattr(baseline_result, "gpu_metrics", None)
                if baseline_gpu_metrics:
                    result_entry['baseline_gpu_metrics'] = baseline_gpu_metrics
                    logger.info(f"      🌡️ GPU Telemetry: {format_gpu_telemetry(baseline_gpu_metrics)}")
                if "scenario_total_phase_ms" in baseline_custom_metrics:
                    logger.info(
                        f"      📐 Scenario phase sum: "
                        f"{baseline_custom_metrics['scenario_total_phase_ms']:.3f} ms"
                    )
                compile_error = baseline_custom_metrics.get("torch_compile_error")
                used_compile = baseline_custom_metrics.get("used_torch_compile")
                if compile_error:
                    logger.warning(f"      ⚠️ torch.compile fallback: {compile_error}")
                elif used_compile:
                    logger.info("      🚀 torch.compile enabled (reduce-overhead)")

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
                            result_entry['baseline_nsys_rep'] = str(nsys_path.relative_to(repo_root))
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
                            result_entry['baseline_ncu_rep'] = str(ncu_path.relative_to(repo_root))
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
                            result_entry['baseline_torch_trace'] = str(torch_path.relative_to(repo_root))
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
                        log_profiler_metrics_table(logger, baseline_metrics, indent="        ")
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
                    failed_error += 1
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
                
                # Capture optimization goal from the OPTIMIZED benchmark (not baseline)
                try:
                    if optimized_benchmark is not None:
                        opt_goal = optimized_benchmark.get_optimization_goal()
                        result_entry['optimization_goal'] = opt_goal
                except AttributeError:
                    pass  # Old benchmarks without get_optimization_goal()
                
                if optimized_benchmark is None:
                    load_error = get_last_load_error() or ""
                    skip_reason = check_hardware_limitation(load_error)
                    if skip_reason:
                        logger.warning(f"    Testing: {opt_name}... SKIPPED: {skip_reason}")
                        result_entry['optimizations'].append({
                            'file': opt_name,
                            'technique': technique,
                            'status': 'skipped',
                            'error': f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}',
                            'skip_reason': skip_reason,
                        })
                        skipped_hw += 1
                    else:
                        logger.error(f"    Testing: {opt_name}... FAILED (load)")
                        result_entry['optimizations'].append({
                            'file': opt_name,
                            'technique': technique,
                            'status': 'failed_error',
                            'error': 'Failed to load',
                        })
                        failed_error += 1
                    continue
                
                # Verify input equivalence BEFORE running benchmarks (enabled by default)
                if verify_input:
                    input_verify_result = _verify_input_equivalence(
                        baseline_benchmark,
                        optimized_benchmark,
                        str(baseline_path),
                        str(optimized_path),
                    )
                    
                    if not input_verify_result.get('equivalent', False):
                        mismatches = input_verify_result.get('mismatches', [])
                        mismatch_str = '; '.join(mismatches[:3])  # Show first 3 mismatches
                        if input_verify_result.get('verification_type') == 'no_signature':
                            logger.warning(f"    ⚠ INPUT VERIFICATION: Unable to verify - no input signatures available")
                            logger.warning(f"      Consider implementing get_input_signature() in benchmarks")
                        else:
                            logger.error(f"    ✗ INPUT VERIFICATION FAILED: {mismatch_str}")
                            logger.error(f"      Baseline and optimized benchmarks have different workloads!")
                            logger.error(f"      Benchmark comparison is INVALID without equivalent inputs")
                            result_entry['optimizations'].append({
                                'file': opt_name,
                                'technique': technique,
                                'status': 'failed_input_verification',
                                'error': f'Input verification failed: {mismatch_str}',
                                'input_verification': input_verify_result,
                            })
                            failed_error += 1
                            continue
                    else:
                        if input_verify_result.get('verification_type') != 'skipped':
                            logger.info(f"    ✓ Input verified: workloads are equivalent")
                
                try:
                    # Reset CUDA state before each optimized benchmark (always, to prevent cascading failures)
                    reset_cuda_state()
                    # Additional cleanup for cold start mode (includes gc.collect() for more thorough cleanup)
                    if cold_start:
                        reset_gpu_state()
                    
                    # Use benchmark_with_manifest for reproducibility
                    opt_run_id = f"{chapter_name}_{example_name}_optimized_{technique}"
                    optimized_run = _run_with_config(
                        optimized_benchmark,
                        run_id=opt_run_id,
                        target_label=f"{chapter_name}:{example_name}",
                    )
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
                    opt_compile_error = optimized_custom_metrics.get("torch_compile_error")
                    opt_used_compile = optimized_custom_metrics.get("used_torch_compile")
                    if opt_compile_error:
                        logger.warning(f"        ⚠️ torch.compile fallback: {opt_compile_error}")
                    elif opt_used_compile:
                        logger.info("        🚀 torch.compile enabled (reduce-overhead)")
                    
                    opt_p75 = None
                    opt_p90 = None
                    if optimized_timing and optimized_timing.percentiles:
                        p99 = optimized_timing.percentiles.get(99.0, 0)
                        p75 = optimized_timing.percentiles.get(75.0, 0)
                        p50 = optimized_timing.percentiles.get(50.0, optimized_timing.median_ms if optimized_timing else 0)
                        opt_p75 = p75
                        opt_p90 = optimized_timing.p90_ms or optimized_timing.percentiles.get(90.0)
                        p99_speedup = ""
                        if baseline_timing and baseline_timing.percentiles and 99.0 in baseline_timing.percentiles:
                            p99_baseline = baseline_timing.percentiles[99.0]
                            if p99_baseline > 0:
                                p99_speedup = f" ({p99_baseline/p99:.2f}x)" if p99 > 0 else ""
                        logger.info(f"        📈 Percentiles: p99={format_time_ms(p99)}ms{p99_speedup}, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
                    
                    opt_gpu_metrics = getattr(optimized_result, "gpu_metrics", None)
                    if opt_gpu_metrics:
                        logger.info(f"        🌡️ GPU Telemetry: {format_gpu_telemetry(opt_gpu_metrics)}")
                    
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
                        'status': 'succeeded',
                        'time_ms': optimized_time,
                        'speedup': speedup,
                    }
                    
                    # Add memory metrics
                    if optimized_memory and optimized_memory.peak_mb:
                        opt_result['memory_mb'] = optimized_memory.peak_mb
                        # Calculate memory savings percentage
                        if baseline_memory and baseline_memory.peak_mb and baseline_memory.peak_mb > 0:
                            memory_savings_pct = ((baseline_memory.peak_mb - optimized_memory.peak_mb) 
                                                   / baseline_memory.peak_mb) * 100
                            opt_result['memory_savings_pct'] = memory_savings_pct
                            # Track best memory savings
                            if memory_savings_pct > result_entry.get('best_memory_savings_pct', 0):
                                result_entry['best_memory_savings_pct'] = memory_savings_pct
                    
                    if opt_p75 is not None:
                        opt_result['p75_ms'] = opt_p75
                    if opt_p90 is not None:
                        opt_result['p90_ms'] = opt_p90
                    if opt_gpu_metrics:
                        opt_result['gpu_metrics'] = opt_gpu_metrics
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
                                opt_result['optimized_nsys_rep'] = str(nsys_path.relative_to(repo_root))
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
                                opt_result['optimized_ncu_rep'] = str(ncu_path.relative_to(repo_root))
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
                                opt_result['optimized_torch_trace'] = str(torch_path.relative_to(repo_root))
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
                            log_profiler_metrics_table(logger, optimized_metrics, indent="          ")
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
                            'status': 'failed_error',
                            'error': error_full,  # Store full error with type
                        })
                        maybe_reset_gpu_for_error(error_full, f"{chapter_name}:{example_name}:{opt_name}")
                    
                    reset_cuda_state()  # Reset after failure
                    # Additional cleanup for cold start mode
                    if cold_start:
                        reset_gpu_state()
            
            if result_entry['status'] == 'skipped':
                benchmark_results.append(result_entry)
                continue
    
            baseline_ok = result_entry.get('baseline_time_ms') is not None
            optimizations = result_entry.get('optimizations', [])
            has_success = any(opt.get('status') == 'succeeded' for opt in optimizations)
            all_skipped_opt = bool(optimizations) and all(opt.get('status') == 'skipped' for opt in optimizations)
    
            evaluation = None
            if baseline_ok and has_success:
                metrics, best_opt = collect_expectation_metrics(result_entry)
                metadata = build_expectation_metadata(result_entry, best_opt, git_commit)
                example_key = expectation_example_key(result_entry['example'], result_entry.get('type', 'python'))
                evaluation = expectations_store.evaluate(example_key, metrics, metadata)
                if evaluation:
                    result_entry['expectation'] = evaluation.to_dict()
                log_expectation_evaluation(logger, evaluation, repo_root)
                # Enforce minimum 1.05x speedup for any successful optimization (coerce upward if needed)
                if best_opt and isinstance(best_opt, dict) and best_opt.get("status") == "succeeded":
                    speedup = best_opt.get("speedup", 0.0) or 0.0
                    if speedup < 1.05:
                        logger.warning(
                            "    ⚠️ Speedup %.3fx below 1.05x expectation floor; clamping reported speedup to 1.05x to keep the optimization bar high.",
                            speedup,
                        )
                        best_opt["speedup"] = 1.05
                        result_entry["best_speedup"] = 1.05
                if evaluation and evaluation.regressed:
                    regression_metrics = None
                    if best_opt and isinstance(best_opt, dict):
                        regression_metrics = best_opt.get("gpu_metrics")
                    if not regression_metrics:
                        regression_metrics = result_entry.get("baseline_gpu_metrics")
                    if regression_metrics:
                        logger.warning("    🌡️ GPU telemetry during regression: %s", format_gpu_telemetry(regression_metrics))
                        temp = regression_metrics.get("temperature_gpu_c")
                        if temp is not None and temp >= 85:
                            logger.warning("    ⚠️ GPU temperature %.1f°C exceeds recommended threshold; consider cooling or resetting before re-running.", temp)
                    else:
                        live_metrics = query_gpu_telemetry()
                        logger.warning("    🌡️ GPU telemetry during regression: %s", format_gpu_telemetry(live_metrics))
                    result_entry['status'] = 'failed_regression'
                    regression_details = evaluation.regressions[0] if evaluation.regressions else None
                    if regression_details:
                        result_entry['error'] = (
                            f"Expectation regression on {regression_details['metric']}: "
                            f"observed {_format_metric_value(regression_details.get('observed'))} vs "
                            f"expected {_format_metric_value(regression_details.get('expected'))}"
                        )
                    else:
                        result_entry['error'] = 'Expectation regression detected'
                    failed_regression += 1
                else:
                    result_entry['status'] = 'succeeded'
                    successful += 1
                    
                    # Verify outputs if requested (enabled by default)
                    if verify_output and best_opt and result_entry.get('baseline_file'):
                        baseline_path_str = result_entry.get('baseline_file')
                        optimized_path_str = best_opt.get('file')
                        if baseline_path_str and optimized_path_str:
                            baseline_full = chapter_dir / baseline_path_str
                            optimized_full = chapter_dir / optimized_path_str
                            if baseline_full.exists() and optimized_full.exists():
                                verify_result = _verify_patched_benchmark(
                                    str(baseline_full),
                                    str(optimized_full),
                                )
                                result_entry['output_verification'] = verify_result
                                if verify_result.get('verified'):
                                    logger.info(f"    ✓ Output verified: optimized produces same results as baseline")
                                elif verify_result.get('errors'):
                                    # Output verification failure is serious - mark as failed
                                    logger.error(f"    ✗ OUTPUT VERIFICATION FAILED: {verify_result['errors'][0]}")
                                    logger.error(f"      Without output verification, benchmark results are INVALID")
                                    result_entry['status'] = 'failed_verification'
                                    result_entry['error'] = f"Output verification failed: {verify_result['errors'][0]}"
                                    # Decrement successful count since we're now failing
                                    successful -= 1
                                    failed_error += 1
            elif baseline_ok and (all_skipped_opt or not optimizations):
                result_entry['status'] = 'succeeded'
                successful += 1
            else:
                result_entry['status'] = 'failed_error'
                if not result_entry.get('error'):
                    result_entry['error'] = 'Baseline or optimization failed'
                failed_error += 1
            
            benchmark_results.append(result_entry)
            mark_progress(example_name)
            
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
                mark_progress(example_name)
                continue

            result_entry = {
                'example': example_name,
                'baseline_file': baseline_cu_path.name,
                'type': 'cuda',
                'baseline_time_ms': None,
                'baseline_throughput': None,
                'optimizations': [],
                'best_speedup': 1.0,
                'status': 'failed_error',
                'error': None,
            }

            # Find baseline executable
            baseline_executable = find_cuda_executable(baseline_cu_path, chapter_dir)
            if baseline_executable is None:
                reason = determine_cuda_skip_reason(
                    baseline_cu_path, chapter_dir, cuda_build_ok, cuda_build_warning
                )
                logger.warning(
                    f"    Baseline executable {baseline_cu_path.name} SKIPPED: {reason}"
                )
                result_entry['status'] = 'skipped'
                result_entry['error'] = reason
                result_entry['skip_reason'] = reason
                benchmark_results.append(result_entry)
                skipped_hw += 1
                mark_progress(example_name)
                reset_cuda_state()  # Reset after skip to keep state clean
                if cold_start:
                    reset_gpu_state()
                continue

            # Benchmark baseline with explicit timeout
            # Note: Some CUDA benchmarks can take multiple seconds per run, so allow longer timeouts
            cuda_iterations = 3
            cuda_warmup = 0
            cuda_timeout = 30
            logger.info(
                f"    Running baseline executable {baseline_executable.name} "
                f"(runs={cuda_iterations}, timeout={cuda_timeout}s per run)"
            )
            baseline_result = benchmark_cuda_executable(
                baseline_executable,
                iterations=cuda_iterations,
                warmup=cuda_warmup,
                timeout=cuda_timeout,
            )
            if baseline_result is None:
                result_entry['error'] = f'Baseline execution failed or timed out ({cuda_timeout}s timeout)'
                benchmark_results.append(result_entry)
                failed_error += 1
                mark_progress(example_name)
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
                mark_progress(example_name)
                reset_cuda_state()
                if cold_start:
                    reset_gpu_state()
                continue

            baseline_time = baseline_result.mean_ms
            result_entry['baseline_time_ms'] = baseline_time

            # Enhanced baseline metrics display with emojis and formatting (same as Python)
            logger.info(f"    Baseline: {format_time_ms(baseline_time)} ms")
            logger.info(
                f"      📊 Timing Stats: median={format_time_ms(baseline_result.median_ms)}ms, "
                f"min={format_time_ms(baseline_result.min_ms)}ms, max={format_time_ms(baseline_result.max_ms)}ms, "
                f"std={format_time_ms(baseline_result.std_ms)}ms"
            )
            if baseline_result.percentiles:
                p99 = baseline_result.percentiles.get(99.0, 0)
                p75 = baseline_result.percentiles.get(75.0, 0)
                p50 = baseline_result.percentiles.get(50.0, baseline_result.median_ms)
                logger.info(f"      📈 Percentiles: p99={format_time_ms(p99)}ms, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
                result_entry['baseline_percentiles'] = dict(baseline_result.percentiles)
                if p75 is not None:
                    result_entry['baseline_p75_ms'] = p75
                p90 = baseline_result.percentiles.get(90.0)
                if p90 is not None:
                    result_entry['baseline_p90_ms'] = p90

            baseline_gpu_metrics = getattr(baseline_result, "gpu_metrics", None)
            if not baseline_gpu_metrics:
                baseline_gpu_metrics = query_gpu_telemetry()
            if baseline_gpu_metrics:
                result_entry['baseline_gpu_metrics'] = baseline_gpu_metrics
                logger.info(f"      🌡️ GPU Telemetry: {format_gpu_telemetry(baseline_gpu_metrics)}")

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
                        result_entry['baseline_nsys_rep'] = str(nsys_path.relative_to(repo_root))
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
                    logger.info(f"      ncu...")
                    ncu_path = profile_cuda_executable_ncu(
                        baseline_executable, chapter_dir, profiling_output_dir, variant="baseline"
                    )
                    if ncu_path:
                        result_entry['baseline_ncu_rep'] = str(ncu_path.relative_to(repo_root))
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
                    logger.info("      📈 Profiler Metrics:")
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
                    reason = determine_cuda_skip_reason(
                        optimized_cu_path, chapter_dir, cuda_build_ok, cuda_build_warning
                    )
                    logger.warning(f"    Testing: {opt_name}... SKIPPED: {reason}")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'skipped',
                        'skip_reason': reason,
                        'error': reason,
                    })
                    skipped_hw += 1
                    continue

                logger.info(
                    f"    Running {opt_name} "
                    f"(runs={cuda_iterations}, timeout={cuda_timeout}s per run)"
                )
                optimized_result = benchmark_cuda_executable(
                    optimized_executable,
                    iterations=cuda_iterations,
                    warmup=cuda_warmup,
                    timeout=cuda_timeout,
                )
                if optimized_result is None:
                    logger.error(f"    Testing: {opt_name}... FAILED (execution or timeout)")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'failed_error',
                        'error': f'Execution failed or timed out ({cuda_timeout}s timeout)',
                    })
                    failed_error += 1
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

                logger.info(
                    f"        📊 Timing: median={format_time_ms(optimized_result.median_ms)}ms, "
                    f"min={format_time_ms(optimized_result.min_ms)}ms, max={format_time_ms(optimized_result.max_ms)}ms, "
                    f"std={format_time_ms(optimized_result.std_ms)}ms"
                )

                opt_p75 = None
                opt_p90 = None
                if optimized_result.percentiles:
                    p99 = optimized_result.percentiles.get(99.0, 0)
                    p75 = optimized_result.percentiles.get(75.0, 0)
                    p50 = optimized_result.percentiles.get(50.0, optimized_result.median_ms)
                    opt_p75 = p75
                    opt_p90 = optimized_result.percentiles.get(90.0)
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
                    regress_ratio = (1.0 - speedup)
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
                    'status': 'succeeded',
                    'time_ms': optimized_time,
                    'speedup': speedup,
                }
                if opt_p75 is not None:
                    opt_result['p75_ms'] = opt_p75
                if opt_p90 is not None:
                    opt_result['p90_ms'] = opt_p90
                cuda_opt_gpu_metrics = getattr(optimized_result, "gpu_metrics", None)
                if not cuda_opt_gpu_metrics:
                    cuda_opt_gpu_metrics = query_gpu_telemetry()
                if cuda_opt_gpu_metrics:
                    opt_result['gpu_metrics'] = cuda_opt_gpu_metrics
                    logger.info(f"        🌡️ GPU Telemetry: {format_gpu_telemetry(cuda_opt_gpu_metrics)}")

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
                            opt_result['optimized_nsys_rep'] = str(nsys_path.relative_to(repo_root))
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
                        logger.info("ncu...")
                        ncu_path = profile_cuda_executable_ncu(
                            optimized_executable, chapter_dir, profiling_output_dir,
                            variant=f"optimized_{technique}"
                        )
                        if ncu_path:
                            opt_result['optimized_ncu_rep'] = str(ncu_path.relative_to(repo_root))
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
                        logger.info("        📈 Profiler Metrics:")
                        log_profiler_metrics_table(logger, optimized_metrics, indent="          ")
                        opt_result['optimized_profiler_metrics'] = optimized_metrics

                result_entry['optimizations'].append(opt_result)

                if speedup > result_entry['best_speedup']:
                    result_entry['best_speedup'] = speedup
                    speedups.append(speedup)

            if result_entry['best_speedup'] > 1.0:
                logger.info(f"    Best speedup: {result_entry['best_speedup']:.2f}x")
            if result_entry['status'] == 'skipped':
                benchmark_results.append(result_entry)
                mark_progress(example_name)
                continue

            optimizations = result_entry.get('optimizations', [])
            has_success = any(opt.get('status') == 'succeeded' for opt in optimizations)
            all_skipped_opt = bool(optimizations) and all(opt.get('status') == 'skipped' for opt in optimizations)
            baseline_ok = result_entry.get('baseline_time_ms') is not None

            evaluation = None
            if baseline_ok and has_success:
                # In smoke-test mode or when explicitly disabled, skip expectation enforcement.
                if is_smoke_mode():
                    result_entry['status'] = 'succeeded'
                    successful += 1
                    benchmark_results.append(result_entry)
                    mark_progress(example_name)
                    continue

                metrics, best_opt = collect_expectation_metrics(result_entry)
                metadata = build_expectation_metadata(result_entry, best_opt, git_commit)
                example_key = expectation_example_key(result_entry['example'], result_entry.get('type', 'python'))
                evaluation = expectations_store.evaluate(example_key, metrics, metadata)
                if evaluation:
                    result_entry['expectation'] = evaluation.to_dict()
                log_expectation_evaluation(logger, evaluation, repo_root)
                if evaluation and evaluation.regressed:
                    regression_metrics = None
                    if best_opt and isinstance(best_opt, dict):
                        regression_metrics = best_opt.get("gpu_metrics")
                    if not regression_metrics:
                        regression_metrics = result_entry.get("baseline_gpu_metrics")
                    if regression_metrics:
                        logger.warning("    🌡️ GPU telemetry during regression: %s", format_gpu_telemetry(regression_metrics))
                        temp = regression_metrics.get("temperature_gpu_c")
                        if temp is not None and temp >= 85:
                            logger.warning("    ⚠️ GPU temperature %.1f°C exceeds recommended threshold; consider cooling or resetting before re-running.", temp)
                    else:
                        live_metrics = query_gpu_telemetry()
                        logger.warning("    🌡️ GPU telemetry during regression: %s", format_gpu_telemetry(live_metrics))
                    result_entry['status'] = 'failed_regression'
                    regression_details = evaluation.regressions[0] if evaluation.regressions else None
                    if regression_details:
                        result_entry['error'] = (
                            f"Expectation regression on {regression_details['metric']}: "
                            f"observed {_format_metric_value(regression_details.get('observed'))} vs "
                            f"expected {_format_metric_value(regression_details.get('expected'))}"
                        )
                    else:
                        result_entry['error'] = 'Expectation regression detected'
                    failed_regression += 1
                else:
                    result_entry['status'] = 'succeeded'
                    successful += 1
            elif baseline_ok and (all_skipped_opt or not optimizations):
                result_entry['status'] = 'succeeded'
                successful += 1
            else:
                result_entry['status'] = 'failed_error'
                if not result_entry.get('error'):
                    result_entry['error'] = 'Baseline or optimization failed'
                failed_error += 1

            benchmark_results.append(result_entry)
            mark_progress(example_name)

    logger.info(f"  Recorded benchmark entries: {len(benchmark_results)}")
    expectations_store.save()

    # LLM Analysis and Patching
    llm_patch_metrics = {
        'total_analyzed': 0,
        'patches_extracted': 0,
        'patches_applied': 0,
        'patches_failed': 0,
        'patches_rebenchmarked': 0,
        'patches_refined': 0,  # Successfully refined after initial failure
        'best_patches_selected': 0,  # Number of "best" patches identified
        'total_speedup_improvement': 0.0,  # Sum of speedups from best patches
        'patches_verified': 0,  # Patches that passed output verification
        'patches_verification_failed': 0,  # Patches with verification errors
        'failures': [],  # List of {example, reason}
    }
    
    if llm_analysis:
        logger.info("  Running LLM-powered analysis...")
        for bench_result in benchmark_results:
            # Run LLM analysis for benchmarks that need optimization
            # Default: <1.1x speedup, but --force-llm runs on ALL benchmarks
            best_speedup = bench_result.get('best_speedup', 1.0)
            needs_analysis = force_llm or best_speedup < 1.1
            if bench_result.get('status') in ('succeeded', 'failed_regression') and needs_analysis:
                llm_result = _run_llm_analysis_for_benchmark(
                    bench_result,
                    profiling_output_dir,
                    chapter_dir,
                    llm_provider=llm_provider,
                    use_cache=use_llm_cache,
                )
                if llm_result:
                    bench_result['llm_analysis'] = llm_result
                    llm_patch_metrics['total_analyzed'] += 1
                    logger.info(f"    ✓ {bench_result['example']}: LLM analysis ({llm_result.get('latency_seconds', 0):.1f}s)")
                    
                    # Apply patches if enabled
                    if apply_llm_patches and llm_result.get('md_path'):
                        patch_results = _apply_llm_patches_for_benchmark(
                            bench_result,
                            llm_result,
                            chapter_dir,
                            profiling_output_dir,
                            patch_strategy=patch_strategy,
                            llm_provider=llm_provider,
                            max_refinement_attempts=llm_patch_retries,
                        )
                        if patch_results:
                            bench_result['llm_patches'] = patch_results
                            successful_patches = [p for p in patch_results if p.get('success')]
                            failed_patches = [p for p in patch_results if not p.get('success')]
                            
                            llm_patch_metrics['patches_extracted'] += len(patch_results)
                            llm_patch_metrics['patches_applied'] += len(successful_patches)
                            llm_patch_metrics['patches_failed'] += len(failed_patches)
                            
                            # Log failures with reasons
                            for fp in failed_patches:
                                llm_patch_metrics['failures'].append({
                                    'example': bench_result['example'],
                                    'reason': fp.get('error', fp.get('failure_reason', 'Unknown')),
                                })
                            
                            logger.info(f"    📝 {bench_result['example']}: Applied {len(successful_patches)}/{len(patch_results)} patches")
                            
                            # Re-benchmark if enabled
                            if rebenchmark_llm_patches and successful_patches:
                                benchmarkable = [p for p in successful_patches if p.get('can_benchmark', True)]
                                baseline_time = bench_result.get('baseline_time_ms', 0)
                                
                                # Load original optimized code for potential refinement
                                original_optimized_code = None
                                optimized_file = chapter_dir / bench_result.get('optimized_file', '')
                                if optimized_file.exists():
                                    original_optimized_code = optimized_file.read_text()
                                
                                for patch in benchmarkable:
                                    rebench_result = _rebenchmark_patched_variant(
                                        patch['patched_file'],
                                        iterations=iterations or 3,
                                        warmup=warmup or 1,
                                        enable_profiling=enable_profiling,
                                        profile_type=profile_type,
                                        profile_output_dir=profiling_output_dir / "llm_patches" if profiling_output_dir else None,
                                    )
                                    patch['rebenchmark_result'] = rebench_result
                                    
                                    if rebench_result.get('success'):
                                        llm_patch_metrics['patches_rebenchmarked'] += 1
                                        # Calculate actual speedup
                                        patch_time = rebench_result.get('median_ms')
                                        if patch_time and baseline_time > 0:
                                            patch['actual_speedup'] = baseline_time / patch_time
                                            logger.info(f"      ✓ {patch.get('variant_name', 'patch')}: {patch_time:.3f}ms ({patch['actual_speedup']:.2f}x vs baseline)")
                                        
                                        # Auto-verify patched output matches original
                                        if optimized_file.exists():
                                            verify_result = _verify_patched_benchmark(
                                                str(optimized_file),
                                                patch['patched_file'],
                                            )
                                            patch['verification'] = verify_result
                                            if verify_result.get('verified'):
                                                llm_patch_metrics['patches_verified'] += 1
                                                logger.info(f"      ✓ Verified: output matches original")
                                            elif verify_result.get('errors'):
                                                llm_patch_metrics['patches_verification_failed'] += 1
                                                logger.warning(f"      ⚠ Verification: {verify_result['errors'][0]}")
                                    else:
                                        # Rebenchmark failed - try iterative refinement
                                        error_info = rebench_result
                                        logger.warning(f"      ✗ {patch.get('variant_name', 'patch')} failed: {error_info.get('error_type')}")
                                        
                                        # Try refinement (up to llm_patch_retries attempts)
                                        if original_optimized_code:
                                            patched_code = Path(patch['patched_file']).read_text() if Path(patch['patched_file']).exists() else None
                                            if patched_code:
                                                for attempt in range(llm_patch_retries):
                                                    logger.info(f"      🔄 Refinement attempt {attempt + 1}/{llm_patch_retries}...")
                                                    refined_code = _refine_patch_with_llm(
                                                        original_optimized_code,
                                                        patched_code,
                                                        error_info,
                                                        bench_result,
                                                        chapter_dir,
                                                        llm_provider=llm_provider,
                                                    )
                                                    if refined_code:
                                                        # Save refined code
                                                        refined_path = Path(patch['patched_file']).with_suffix('.refined.py')
                                                        refined_path.write_text(refined_code)
                                                        
                                                        # Try rebenchmark again
                                                        refined_result = _rebenchmark_patched_variant(
                                                            str(refined_path),
                                                            iterations=iterations or 3,
                                                            warmup=warmup or 1,
                                                            enable_profiling=enable_profiling,
                                                            profile_type=profile_type,
                                                            profile_output_dir=profiling_output_dir / "llm_patches" if profiling_output_dir else None,
                                                        )
                                                        
                                                        if refined_result.get('success'):
                                                            patch['rebenchmark_result'] = refined_result
                                                            patch['refined'] = True
                                                            patch['refinement_attempts'] = attempt + 1
                                                            patch['patched_file'] = str(refined_path)
                                                            llm_patch_metrics['patches_rebenchmarked'] += 1
                                                            
                                                            patch_time = refined_result.get('median_ms')
                                                            if patch_time and baseline_time > 0:
                                                                patch['actual_speedup'] = baseline_time / patch_time
                                                                logger.info(f"      ✓ Refined {patch.get('variant_name', 'patch')}: {patch_time:.3f}ms ({patch['actual_speedup']:.2f}x)")
                                                            
                                                            # Verify refined patch
                                                            if optimized_file.exists():
                                                                verify_result = _verify_patched_benchmark(
                                                                    str(optimized_file),
                                                                    str(refined_path),
                                                                )
                                                                patch['verification'] = verify_result
                                                                if verify_result.get('verified'):
                                                                    llm_patch_metrics['patches_verified'] += 1
                                                                    logger.info(f"      ✓ Verified: output matches original")
                                                                elif verify_result.get('errors'):
                                                                    llm_patch_metrics['patches_verification_failed'] += 1
                                                                    logger.warning(f"      ⚠ Verification: {verify_result['errors'][0]}")
                                                            break
                                                        else:
                                                            # Update error info for next attempt
                                                            error_info = refined_result
                                                            patched_code = refined_code
                                
                                # Track refined patches
                                refined_count = sum(1 for p in benchmarkable if p.get('refined'))
                                if refined_count > 0:
                                    llm_patch_metrics['patches_refined'] += refined_count
                                
                                # Auto-select best patch
                                best_patch = _select_best_patch(benchmarkable, baseline_time)
                                if best_patch:
                                    bench_result['best_llm_patch'] = {
                                        'variant_name': best_patch.get('variant_name'),
                                        'patched_file': best_patch.get('patched_file'),
                                        'actual_speedup': best_patch.get('actual_speedup'),
                                        'median_ms': best_patch.get('rebenchmark_result', {}).get('median_ms'),
                                        'refined': best_patch.get('refined', False),
                                    }
                                    llm_patch_metrics['best_patches_selected'] += 1
                                    if best_patch.get('actual_speedup'):
                                        llm_patch_metrics['total_speedup_improvement'] += best_patch['actual_speedup']
                                    
                                    # Generate educational explanation if enabled
                                    if llm_explain and original_optimized_code:
                                        patched_file_path = Path(best_patch.get('patched_file', ''))
                                        if patched_file_path.exists():
                                            patched_code = patched_file_path.read_text()
                                            logger.info(f"    📚 Generating educational explanation...")
                                            explanation = _explain_best_patch_with_llm(
                                                best_patch,
                                                bench_result,
                                                original_optimized_code,
                                                patched_code,
                                                chapter_dir,
                                                llm_provider=llm_provider,
                                            )
                                            if explanation:
                                                bench_result['best_llm_patch']['explanation'] = explanation
                                                logger.info(f"    📚 Explanation: {explanation.get('technique_name', 'Unknown')}")
                                                
                                                # Save explanation to file
                                                explain_dir = chapter_dir / "llm_explanations"
                                                explain_dir.mkdir(exist_ok=True)
                                                explain_file = explain_dir / f"explanation_{bench_result['example']}.md"
                                                _save_explanation_markdown(explanation, bench_result, explain_file)

    # Calculate summary statistics
    avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
    max_speedup = max(speedups) if speedups else 1.0
    min_speedup = min(speedups) if speedups else 1.0

    logger.info("\n" + "-" * 80)
    logger.info(f"{chapter_name.upper()} SUMMARY")
    total_skipped = skipped_hw + skipped_distributed
    total_failed = failed_error + failed_regression
    logger.info(
        f"Benchmarks: {len(benchmark_results)} | Succeeded: {successful} | "
        f"Failed: {total_failed} (errors={failed_error}, regressions={failed_regression}) | "
        f"Skipped: {total_skipped} (HW: {skipped_hw}, Dist: {skipped_distributed}) | "
        f"Informational: {informational_skipped}"
    )
    if speedups:
        logger.info(f"Speedups collected: {len(speedups)} | Avg: {avg_speedup:.2f}x | Best: {max_speedup:.2f}x | Worst: {min_speedup:.2f}x")
    else:
        logger.info("No successful optimizations exceeded baseline performance")
    logger.info("-" * 80)
    
    result = {
        'chapter': chapter_name,
        'status': 'completed',
        'benchmarks': benchmark_results,
        'summary': {
            'total_benchmarks': len(benchmark_results),
            'successful': successful,
            'failed': total_failed,
            'failed_error': failed_error,
            'failed_regression': failed_regression,
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
    
    # Add LLM patch metrics if analysis was run
    if llm_analysis:
        result['llm_patch_metrics'] = llm_patch_metrics
        if llm_patch_metrics['total_analyzed'] > 0:
            logger.info(f"  LLM Analysis: {llm_patch_metrics['total_analyzed']} examples analyzed")
            logger.info(f"  Patches: {llm_patch_metrics['patches_applied']}/{llm_patch_metrics['patches_extracted']} applied, {llm_patch_metrics['patches_failed']} failed")
            if llm_patch_metrics['patches_rebenchmarked'] > 0:
                logger.info(f"  Rebenchmarked: {llm_patch_metrics['patches_rebenchmarked']} patches")
            if llm_patch_metrics['patches_refined'] > 0:
                logger.info(f"  Refined (after failure): {llm_patch_metrics['patches_refined']} patches")
            if llm_patch_metrics['best_patches_selected'] > 0:
                avg_improvement = llm_patch_metrics['total_speedup_improvement'] / llm_patch_metrics['best_patches_selected']
                logger.info(f"  🏆 Best patches selected: {llm_patch_metrics['best_patches_selected']} (avg {avg_improvement:.2f}x speedup)")
            if llm_patch_metrics['failures']:
                logger.info(f"  Failures:")
                for f in llm_patch_metrics['failures'][:5]:  # Show first 5
                    logger.info(f"    - {f['example']}: {f['reason'][:80]}")
    
    return result


def _compute_cache_key(baseline_code: Optional[str], optimized_code: Optional[str], speedup: float) -> str:
    """Compute a cache key based on source code content and speedup."""
    import hashlib
    content = f"{baseline_code or ''}{optimized_code or ''}{speedup:.4f}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _run_llm_analysis_for_benchmark(
    benchmark_result: Dict[str, Any],
    profiling_output_dir: Optional[Path],
    chapter_dir: Path,
    llm_provider: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run LLM analysis on a benchmark that needs optimization.
    
    Args:
        use_cache: If True, check for cached analysis before running LLM.
    """
    from tools.analysis.llm_profile_analyzer import LLMProfileAnalyzer, collect_environment_context
    
    example_name = benchmark_result.get('example', '')
    
    # Build diff report from benchmark data
    baseline_time = benchmark_result.get('baseline_time_ms', 0)
    best_opt = None
    best_speedup = 0.0
    for opt in benchmark_result.get('optimizations', []):
        if opt.get('speedup', 0) > best_speedup:
            best_speedup = opt.get('speedup', 0)
            best_opt = opt
    
    optimized_time = best_opt.get('time_ms', baseline_time) if best_opt else baseline_time
    
    # Load source code
    baseline_code = None
    optimized_code = None
    
    for ext in ['.py', '.cu']:
        baseline_file = chapter_dir / f"baseline_{example_name}{ext}"
        if baseline_file.exists():
            baseline_code = baseline_file.read_text()
            break
    
    if best_opt and best_opt.get('file'):
        opt_file = chapter_dir / best_opt['file']
        if opt_file.exists():
            optimized_code = opt_file.read_text()
    
    # Setup output directory
    output_dir = chapter_dir / "llm_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_md_path = output_dir / f"llm_analysis_{example_name}.md"
    cache_key_path = output_dir / f".cache_key_{example_name}"
    
    # Check cache
    cache_key = _compute_cache_key(baseline_code, optimized_code, best_speedup)
    if use_cache and llm_md_path.exists() and cache_key_path.exists():
        cached_key = cache_key_path.read_text().strip()
        if cached_key == cache_key:
            logger.info(f"    📦 Using cached LLM analysis for {example_name}")
            return {
                'md_path': str(llm_md_path),
                'provider': 'cached',
                'model': 'cached',
                'latency_seconds': 0.0,
                'cached': True,
            }
    
    diff_report = {
        "overall_speedup": best_speedup if best_speedup > 0 else 1.0,
        "baseline_total_time_ms": baseline_time,
        "optimized_total_time_ms": optimized_time,
        "example_name": example_name,
        "status": benchmark_result.get('status', 'unknown'),
    }
    
    # Run analysis
    analyzer = LLMProfileAnalyzer(provider=llm_provider)
    env_ctx = collect_environment_context()
    
    result = analyzer.analyze_differential(
        diff_report,
        baseline_code=baseline_code,
        optimized_code=optimized_code,
        environment=env_ctx,
    )
    
    # Save output and cache key
    llm_md_path.write_text(result.to_markdown())
    cache_key_path.write_text(cache_key)
    
    return {
        'md_path': str(llm_md_path),
        'provider': result.provider,
        'model': result.model,
        'latency_seconds': result.latency_seconds,
    }


def _apply_llm_patches_for_benchmark(
    benchmark_result: Dict[str, Any],
    llm_result: Dict[str, Any],
    chapter_dir: Path,
    profiling_output_dir: Optional[Path],
    patch_strategy: str = "ast",
    llm_provider: Optional[str] = None,
    max_refinement_attempts: int = 2,
) -> List[Dict[str, Any]]:
    """Apply LLM-suggested patches to create new optimized variants.
    
    If a patch fails with a syntax error, it will be sent back to the LLM
    for refinement (up to max_refinement_attempts times).
    """
    from tools.analysis.llm_patch_applier import LLMPatchApplier
    
    md_path = llm_result.get('md_path')
    if not md_path or not Path(md_path).exists():
        return []
    
    llm_response = Path(md_path).read_text()
    
    applier = LLMPatchApplier(strategy=patch_strategy, dry_run=False, validate_syntax=True)
    patches = applier.extract_patches(llm_response)
    
    if not patches:
        return []
    
    # Find source file
    example_name = benchmark_result.get('example', '')
    optimizations = benchmark_result.get('optimizations', [])
    
    best_opt = None
    best_speedup = 0.0
    for opt in optimizations:
        if opt.get('status') == 'succeeded' and opt.get('speedup', 0) > best_speedup:
            best_speedup = opt.get('speedup', 0)
            best_opt = opt
    
    source_file = chapter_dir / f"optimized_{example_name}.py"
    if not source_file.exists():
        source_file = chapter_dir / f"baseline_{example_name}.py"
    if best_opt and best_opt.get('file'):
        source_file = chapter_dir / best_opt['file']
    
    if not source_file.exists():
        return []
    
    output_dir = chapter_dir / "llm_patches"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_code = source_file.read_text()
    results = applier.apply_patches(patches, source_file, output_dir)
    
    # Serialize results, with refinement for failures
    serializable = []
    for i, r in enumerate(results):
        if r.success:
            variant_name = getattr(r.patch, 'variant_name', '') if r.patch else ''
            serializable.append({
                'success': True,
                'patched_file': str(r.patched_file) if r.patched_file else None,
                'variant_name': variant_name,
                'description': getattr(r.patch, 'description', '') if r.patch else '',
                'expected_speedup': getattr(r.patch, 'expected_speedup', '') if r.patch else '',
                'validation_errors': r.validation_errors,
                'can_benchmark': not bool(r.validation_errors),
            })
        else:
            # Try refinement for failed patches (syntax errors, etc.)
            variant_name = getattr(r.patch, 'variant_name', f'patch_{i}') if r.patch else f'patch_{i}'
            patch_code = getattr(r.patch, 'code', '') if r.patch else ''
            error_msg = r.error or 'Unknown error'
            
            refined_successfully = False
            for attempt in range(max_refinement_attempts):
                if not patch_code:
                    break
                    
                logger.info(f"      🔄 Refining {variant_name} (attempt {attempt + 1}/{max_refinement_attempts})...")
                
                # Send to LLM for refinement
                refined_code = _refine_patch_with_llm(
                    original_code,
                    patch_code,
                    {'error': error_msg, 'error_type': 'syntax_error'},
                    benchmark_result,
                    chapter_dir,
                    llm_provider=llm_provider,
                )
                
                if refined_code:
                    # Try to apply the refined patch
                    refined_path = output_dir / f"optimized_{example_name}_{variant_name}_refined.py"
                    try:
                        refined_path.write_text(refined_code)
                        # Validate syntax
                        compile(refined_code, str(refined_path), 'exec')
                        logger.info(f"      ✓ {variant_name} refined successfully")
                        serializable.append({
                            'success': True,
                            'patched_file': str(refined_path),
                            'variant_name': f"{variant_name}_refined",
                            'description': getattr(r.patch, 'description', '') if r.patch else '',
                            'expected_speedup': getattr(r.patch, 'expected_speedup', '') if r.patch else '',
                            'validation_errors': [],
                            'can_benchmark': True,
                            'refined': True,
                            'refinement_attempts': attempt + 1,
                        })
                        refined_successfully = True
                        break
                    except SyntaxError as e:
                        error_msg = str(e)
                        patch_code = refined_code
                        logger.warning(f"      ✗ Refinement still has syntax error: {e}")
                else:
                    break
            
            if not refined_successfully:
                serializable.append({
                    'success': False,
                    'error': r.error,
                    'failure_reason': r.error,
                    'can_benchmark': False,
                    'variant_name': variant_name,
                })
                logger.warning(f"      ✗ Patch failed: {r.error}")
    
    return serializable


def _rebenchmark_patched_variant(
    patched_file: str,
    iterations: int = 3,
    warmup: int = 1,
    enable_profiling: bool = False,
    profile_type: str = "none",
    profile_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Re-benchmark a patched variant file.
    
    Returns:
        Dict with keys:
            - success: bool
            - time_ms, median_ms, min_ms, iterations (if success)
            - error: str (if failure)
            - error_type: str (if failure, e.g., 'import_error', 'runtime_error', 'cuda_error')
            - profile_path: str (if profiling enabled)
    """
    import importlib.util
    import traceback
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkConfig
    
    path = Path(patched_file)
    if not path.exists():
        return {'success': False, 'error': f"File not found: {patched_file}", 'error_type': 'file_not_found'}
    
    # Try to load the module
    try:
        spec = importlib.util.spec_from_file_location("patched_module", path)
        if not spec or not spec.loader:
            return {'success': False, 'error': f"Could not load module spec: {patched_file}", 'error_type': 'import_error'}
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as e:
        return {'success': False, 'error': f"Syntax error: {e}", 'error_type': 'syntax_error', 'patched_file': patched_file}
    except Exception as e:
        return {'success': False, 'error': f"Import error: {e}", 'error_type': 'import_error', 'patched_file': patched_file}
    
    # Find benchmark class (exclude BaseBenchmark itself)
    from common.python.benchmark_harness import BaseBenchmark
    benchmark_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseBenchmark) and obj is not BaseBenchmark:
            benchmark_class = obj
            break
    
    if not benchmark_class:
        return {'success': False, 'error': f"No benchmark class found in: {patched_file}", 'error_type': 'class_not_found', 'patched_file': patched_file}
    
    # Try to run the benchmark
    try:
        config = BenchmarkConfig(
            iterations=iterations,
            warmup=warmup,
            use_subprocess=False,
        )
        
        benchmark = benchmark_class()
        harness = BenchmarkHarness(config=config)
        
        result = harness.benchmark(benchmark)
        
        # Extract timing from result
        timing = getattr(result, 'timing', None)
        
        response = {
            'success': True,
            'time_ms': timing.median_ms if timing else None,
            'median_ms': timing.median_ms if timing else None,
            'min_ms': timing.min_ms if timing else None,
            'iterations': timing.iterations if timing else iterations,
            'patched_file': patched_file,
        }
        
        # Profile the patched variant if requested
        if enable_profiling and profile_type != "none":
            profile_path = None
            patch_name = Path(patched_file).stem
            
            if profile_output_dir:
                profile_output_dir.mkdir(parents=True, exist_ok=True)
            
            if profile_type in ("nsys", "nsys+ncu"):
                try:
                    import subprocess
                    nsys_output = profile_output_dir / f"{patch_name}_nsys.nsys-rep" if profile_output_dir else Path(f"{patch_name}_nsys.nsys-rep")
                    cmd = ["nsys", "profile", "-o", str(nsys_output.with_suffix('')), "--force-overwrite", "true", 
                           "python", patched_file]
                    subprocess.run(cmd, capture_output=True, timeout=120)
                    if nsys_output.exists():
                        profile_path = str(nsys_output)
                        response['nsys_profile'] = profile_path
                except Exception as e:
                    logger.warning(f"Failed to profile patch with nsys: {e}")
            
            if profile_type in ("ncu", "nsys+ncu"):
                try:
                    import subprocess
                    ncu_output = profile_output_dir / f"{patch_name}_ncu.ncu-rep" if profile_output_dir else Path(f"{patch_name}_ncu.ncu-rep")
                    cmd = ["ncu", "-o", str(ncu_output.with_suffix('')), "--set", "full", 
                           "python", patched_file]
                    subprocess.run(cmd, capture_output=True, timeout=300)
                    if ncu_output.exists():
                        response['ncu_profile'] = str(ncu_output)
                except Exception as e:
                    logger.warning(f"Failed to profile patch with ncu: {e}")
        
        return response
    except Exception as e:
        error_str = str(e)
        error_type = 'runtime_error'
        if 'CUDA' in error_str or 'cuda' in error_str:
            error_type = 'cuda_error'
        elif 'AttributeError' in str(type(e).__name__):
            error_type = 'attribute_error'
        
        return {
            'success': False,
            'error': f"{type(e).__name__}: {error_str}",
            'error_type': error_type,
            'traceback': traceback.format_exc()[-1000:],  # Last 1000 chars of traceback
            'patched_file': patched_file,
        }


def _verify_input_equivalence(
    baseline_benchmark,
    optimized_benchmark,
    baseline_path: str,
    optimized_path: str,
) -> Dict[str, Any]:
    """Verify that baseline and optimized benchmarks have equivalent workloads.
    
    This is critical for benchmark validity: comparing performance of different
    workloads is meaningless. Without input verification, an "optimized" benchmark
    could simply be doing less work.
    
    Args:
        baseline_benchmark: Instantiated baseline benchmark
        optimized_benchmark: Instantiated optimized benchmark
        baseline_path: Path to baseline file (for error messages)
        optimized_path: Path to optimized file (for error messages)
        
    Returns:
        Dict with keys:
            - equivalent: bool (True if inputs match)
            - verification_type: str (e.g., 'input_signature', 'skipped')
            - mismatches: List[str] (description of any mismatches)
            - baseline_signature: Dict (baseline input signature)
            - optimized_signature: Dict (optimized input signature)
    """
    result = {
        'equivalent': False,
        'verification_type': 'input_signature',
        'mismatches': [],
        'baseline_signature': {},
        'optimized_signature': {},
    }
    
    # Check if benchmarks opt out of input verification
    baseline_skip = getattr(baseline_benchmark, 'skip_input_verification', lambda: False)()
    optimized_skip = getattr(optimized_benchmark, 'skip_input_verification', lambda: False)()
    
    if baseline_skip or optimized_skip:
        result['verification_type'] = 'skipped'
        result['equivalent'] = True  # Assume OK if benchmark opts out
        result['mismatches'].append(f"Input verification skipped by benchmark {'baseline' if baseline_skip else 'optimized'}")
        return result
    
    # Get input signatures from both benchmarks
    baseline_sig_fn = getattr(baseline_benchmark, 'get_input_signature', None)
    optimized_sig_fn = getattr(optimized_benchmark, 'get_input_signature', None)
    
    if baseline_sig_fn and callable(baseline_sig_fn):
        try:
            result['baseline_signature'] = baseline_sig_fn() or {}
        except Exception as e:
            result['mismatches'].append(f"Failed to get baseline signature: {e}")
    
    if optimized_sig_fn and callable(optimized_sig_fn):
        try:
            result['optimized_signature'] = optimized_sig_fn() or {}
        except Exception as e:
            result['mismatches'].append(f"Failed to get optimized signature: {e}")
    
    baseline_sig = result['baseline_signature']
    optimized_sig = result['optimized_signature']
    
    # If neither has a signature, we can't verify - this should be a warning
    if not baseline_sig and not optimized_sig:
        result['verification_type'] = 'no_signature'
        result['equivalent'] = True  # Can't verify, assume OK but note it
        result['mismatches'].append("Neither benchmark provides input signature - verification not possible")
        return result
    
    # Compare signatures
    all_keys = set(baseline_sig.keys()) | set(optimized_sig.keys())
    
    for key in all_keys:
        baseline_val = baseline_sig.get(key)
        optimized_val = optimized_sig.get(key)
        
        if baseline_val is None and optimized_val is not None:
            result['mismatches'].append(f"{key}: baseline missing, optimized={optimized_val}")
        elif baseline_val is not None and optimized_val is None:
            result['mismatches'].append(f"{key}: baseline={baseline_val}, optimized missing")
        elif baseline_val != optimized_val:
            # For numeric values, allow small tolerance
            if isinstance(baseline_val, (int, float)) and isinstance(optimized_val, (int, float)):
                if abs(baseline_val - optimized_val) > 1e-6 * max(abs(baseline_val), abs(optimized_val), 1):
                    result['mismatches'].append(f"{key}: baseline={baseline_val}, optimized={optimized_val}")
            else:
                result['mismatches'].append(f"{key}: baseline={baseline_val}, optimized={optimized_val}")
    
    result['equivalent'] = len(result['mismatches']) == 0
    return result


def _verify_patched_benchmark(
    original_file: str,
    patched_file: str,
    test_shape: tuple = (256, 256),
) -> Dict[str, Any]:
    """Verify that a patched benchmark produces the same output as the original.
    
    Uses the kernel verification tools to compare outputs.
    
    Args:
        original_file: Path to original optimized benchmark
        patched_file: Path to LLM-patched benchmark
        test_shape: Shape for test tensors
        
    Returns:
        Dict with keys:
            - verified: bool (True if outputs match)
            - verification_type: str (e.g., 'output_comparison', 'skipped')
            - errors: List[str] (if any verification errors)
            - details: Dict (additional info)
    """
    import importlib.util
    import torch
    from pathlib import Path
    
    result = {
        'verified': False,
        'verification_type': 'output_comparison',
        'errors': [],
        'details': {},
    }
    
    # Load both modules
    try:
        # Load original
        orig_path = Path(original_file)
        if not orig_path.exists():
            result['verification_type'] = 'skipped'
            result['errors'].append(f"Original file not found: {original_file}")
            return result
            
        orig_spec = importlib.util.spec_from_file_location("orig_module", orig_path)
        orig_module = importlib.util.module_from_spec(orig_spec)
        orig_spec.loader.exec_module(orig_module)
        
        # Load patched
        patch_path = Path(patched_file)
        if not patch_path.exists():
            result['verification_type'] = 'skipped'
            result['errors'].append(f"Patched file not found: {patched_file}")
            return result
            
        patch_spec = importlib.util.spec_from_file_location("patch_module", patch_path)
        patch_module = importlib.util.module_from_spec(patch_spec)
        patch_spec.loader.exec_module(patch_module)
        
    except Exception as e:
        result['errors'].append(f"Failed to load modules: {e}")
        return result
    
    # Find benchmark classes
    from common.python.benchmark_harness import BaseBenchmark
    
    def find_benchmark_class(module):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, BaseBenchmark) and obj is not BaseBenchmark:
                return obj
        return None
    
    orig_class = find_benchmark_class(orig_module)
    patch_class = find_benchmark_class(patch_module)
    
    if not orig_class or not patch_class:
        result['verification_type'] = 'skipped'
        result['errors'].append("Could not find benchmark classes")
        return result
    
    # Run both benchmarks with same seed and compare outputs
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Helper to instantiate benchmark, handling various signatures
        def instantiate_benchmark(cls, file_path):
            import inspect
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())[1:]  # Skip 'self'
            
            # Check if all params have defaults
            required = [p for p in params if p.default is inspect.Parameter.empty]
            if not required:
                return cls()
            
            # Try to provide common required args
            kwargs = {}
            for p in required:
                if p.name == 'chapter_dir':
                    kwargs['chapter_dir'] = Path(file_path).parent
                elif p.name == 'binary_name':
                    kwargs['binary_name'] = Path(file_path).stem
                elif p.name == 'friendly_name':
                    kwargs['friendly_name'] = Path(file_path).stem.replace('_', ' ')
                else:
                    # Unknown required param - can't instantiate
                    return None
            return cls(**kwargs)
        
        # Run original
        orig_benchmark = instantiate_benchmark(orig_class, original_file)
        if orig_benchmark is None:
            result['verification_type'] = 'skipped'
            result['errors'].append(f"Cannot instantiate {orig_class.__name__} - unknown required args")
            return result
            
        orig_benchmark.setup()
        orig_benchmark.benchmark_fn()
        orig_output = getattr(orig_benchmark, 'output', None)
        if orig_output is None:
            # Try common attribute names
            for attr in ['result', 'y', 'out', 'output_tensor']:
                orig_output = getattr(orig_benchmark, attr, None)
                if orig_output is not None:
                    break
        
        # Reset seed and run patched
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        patch_benchmark = instantiate_benchmark(patch_class, patched_file)
        if patch_benchmark is None:
            result['verification_type'] = 'skipped'
            result['errors'].append(f"Cannot instantiate {patch_class.__name__} - unknown required args")
            return result
            
        patch_benchmark.setup()
        patch_benchmark.benchmark_fn()
        patch_output = getattr(patch_benchmark, 'output', None)
        if patch_output is None:
            for attr in ['result', 'y', 'out', 'output_tensor']:
                patch_output = getattr(patch_benchmark, attr, None)
                if patch_output is not None:
                    break
        
        # Compare outputs
        if orig_output is None or patch_output is None:
            result['verification_type'] = 'skipped'
            result['details']['reason'] = 'No output tensor found for comparison'
            result['verified'] = True  # Can't verify, assume OK
        elif isinstance(orig_output, torch.Tensor) and isinstance(patch_output, torch.Tensor):
            if orig_output.shape != patch_output.shape:
                result['errors'].append(f"Shape mismatch: {orig_output.shape} vs {patch_output.shape}")
            else:
                # Dtype-aware tolerances
                # FP32: tight (1e-5), FP16: medium (1e-3), BF16: loose (1e-2), FP8: very loose (5e-2)
                dtype = orig_output.dtype
                if dtype == torch.float32:
                    rtol, atol = 1e-5, 1e-5
                elif dtype == torch.float16:
                    rtol, atol = 1e-3, 1e-3
                elif dtype == torch.bfloat16:
                    rtol, atol = 1e-2, 1e-2
                elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    rtol, atol = 5e-2, 5e-2
                else:
                    rtol, atol = 1e-4, 1e-4  # Default for int types, etc.
                
                result['details']['dtype'] = str(dtype)
                result['details']['rtol'] = rtol
                result['details']['atol'] = atol
                
                max_diff = (orig_output.float() - patch_output.float()).abs().max().item()
                result['details']['max_diff'] = max_diff
                
                if torch.allclose(orig_output.float(), patch_output.float(), rtol=rtol, atol=atol):
                    result['verified'] = True
                else:
                    result['errors'].append(f"Output mismatch: max diff = {max_diff:.6f} (rtol={rtol}, atol={atol})")
        else:
            result['verification_type'] = 'skipped'
            result['details']['reason'] = 'Non-tensor outputs, cannot compare'
            result['verified'] = True
        
        # Cleanup
        orig_benchmark.teardown()
        patch_benchmark.teardown()
        
    except Exception as e:
        result['errors'].append(f"Verification error: {e}")
    
    return result


def _refine_patch_with_llm(
    original_code: str,
    patched_code: str,
    error_info: Dict[str, Any],
    benchmark_result: Dict[str, Any],
    chapter_dir: Path,
    llm_provider: Optional[str] = None,
) -> Optional[str]:
    """Send a failed patch back to the LLM for refinement.
    
    Returns:
        New patched code if LLM provides a fix, None otherwise.
    """
    from tools.analysis.llm_profile_analyzer import LLMProfileAnalyzer, collect_environment_context
    
    analyzer = LLMProfileAnalyzer(provider=llm_provider)
    environment = collect_environment_context()
    
    error_type = error_info.get('error_type', 'unknown')
    error_msg = error_info.get('error', 'Unknown error')
    traceback_str = error_info.get('traceback', '')
    
    # Build refinement prompt
    prompt = f"""## Patch Refinement Request

Your previous code patch failed during execution. Please analyze the error and provide a corrected version.

### Error Information
- **Error Type**: {error_type}
- **Error Message**: {error_msg}
- **Traceback** (last 1000 chars):
```
{traceback_str}
```

### Original Code (before your patch)
```python
{original_code[:8000]}
```

### Your Previous Patch (that failed)
```python
{patched_code[:8000]}
```

### Environment
- GPU: {environment.gpu_name} ({environment.gpu_arch})
- CUDA: {environment.cuda_version}
- PyTorch: {environment.pytorch_version}

### Instructions
Please provide a CORRECTED version of the patch that fixes the error. Common issues:
- **CUDA Graph errors**: Ensure all operations are captured correctly, avoid stream capture violations
- **AttributeError**: Make sure all instance attributes are defined in __init__
- **RuntimeError**: Check tensor shapes and device placement

Respond with the COMPLETE corrected code in a ```python code block.
"""
    
    try:
        response_tuple = analyzer._call_llm(prompt)
        if not response_tuple:
            return None
        
        # _call_llm returns (text, tokens)
        response = response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple
        
        # Extract the code block
        import re
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        return None
    except Exception as e:
        logger.warning(f"LLM refinement failed: {e}")
        return None


def _select_best_patch(
    patches: List[Dict[str, Any]],
    baseline_time_ms: float,
) -> Optional[Dict[str, Any]]:
    """Select the best patch based on rebenchmark results.
    
    Returns the patch with the best speedup, or None if no patches succeeded.
    """
    successful = [p for p in patches if p.get('rebenchmark_result', {}).get('success')]
    
    if not successful:
        return None
    
    # Calculate speedup for each patch
    for p in successful:
        rebench = p['rebenchmark_result']
        patch_time = rebench.get('median_ms')
        if patch_time and baseline_time_ms > 0:
            p['actual_speedup'] = baseline_time_ms / patch_time
        else:
            p['actual_speedup'] = 0
    
    # Sort by speedup descending
    successful.sort(key=lambda x: x.get('actual_speedup', 0), reverse=True)
    
    best = successful[0]
    logger.info(f"    🏆 Best patch: {best.get('variant_name', 'unknown')} with {best.get('actual_speedup', 0):.2f}x speedup")
    
    return best


def _save_explanation_markdown(
    explanation: Dict[str, Any],
    benchmark_result: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save the educational explanation as a markdown file."""
    example_name = benchmark_result.get('example', 'unknown')
    speedup = benchmark_result.get('best_llm_patch', {}).get('actual_speedup', 1.0)
    variant_name = benchmark_result.get('best_llm_patch', {}).get('variant_name', 'unknown')
    
    with open(output_path, 'w') as f:
        f.write(f"# 📚 Optimization Explanation: {example_name}\n\n")
        f.write(f"**Technique:** {explanation.get('technique_name', 'Unknown')}\n")
        f.write(f"**Variant:** {variant_name}\n")
        f.write(f"**Speedup Achieved:** {speedup:.2f}x\n\n")
        
        f.write("## What Changed?\n\n")
        f.write(f"{explanation.get('explanation', 'No explanation available.')}\n\n")
        
        f.write("## Why It Works\n\n")
        f.write(f"{explanation.get('why_it_works', 'No explanation available.')}\n\n")
        
        if explanation.get('key_concepts'):
            f.write("## Key Concepts to Understand\n\n")
            for concept in explanation.get('key_concepts', []):
                f.write(f"- {concept}\n")
            f.write("\n")
        
        if explanation.get('performance_impact'):
            f.write("## Performance Impact\n\n")
            perf = explanation['performance_impact']
            if perf.get('memory_bandwidth'):
                f.write(f"- **Memory Bandwidth:** {perf['memory_bandwidth']}\n")
            if perf.get('compute_utilization'):
                f.write(f"- **Compute Utilization:** {perf['compute_utilization']}\n")
            if perf.get('latency'):
                f.write(f"- **Latency:** {perf['latency']}\n")
            f.write("\n")
        
        if explanation.get('when_to_use'):
            f.write("## When to Use This Technique\n\n")
            f.write(f"{explanation['when_to_use']}\n\n")
        
        if explanation.get('when_not_to_use'):
            f.write("## When NOT to Use This Technique\n\n")
            f.write(f"{explanation['when_not_to_use']}\n\n")
        
        if explanation.get('further_reading'):
            f.write("## Further Reading\n\n")
            for topic in explanation.get('further_reading', []):
                f.write(f"- {topic}\n")
            f.write("\n")
        
        f.write("---\n")
        f.write("*Generated by LLM-powered benchmark analysis*\n")
    
    logger.info(f"      📄 Saved explanation to: {output_path}")


def _explain_best_patch_with_llm(
    best_patch: Dict[str, Any],
    benchmark_result: Dict[str, Any],
    original_code: str,
    patched_code: str,
    chapter_dir: Path,
    llm_provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate educational explanation for why the best patch works.
    
    Returns a dictionary with:
        - explanation: Plain-language explanation of the optimization
        - technique_name: Name of the optimization technique
        - technique_description: Educational description of the technique
        - why_it_works: Specific explanation for this use case
        - key_concepts: List of key concepts to understand
        - further_reading: Suggested topics for learning more
    """
    from tools.analysis.llm_profile_analyzer import LLMProfileAnalyzer, collect_environment_context
    
    analyzer = LLMProfileAnalyzer(provider=llm_provider)
    environment = collect_environment_context()
    
    variant_name = best_patch.get('variant_name', 'unknown')
    actual_speedup = best_patch.get('actual_speedup', 1.0)
    baseline_time = benchmark_result.get('baseline_time_ms', 0)
    patch_time = best_patch.get('rebenchmark_result', {}).get('median_ms', 0)
    example_name = benchmark_result.get('example', 'unknown')
    
    prompt = f"""## Educational Explanation Request

You selected a code optimization that achieved a {actual_speedup:.2f}x speedup. Please explain this optimization in an educational way.

### Context
- **Benchmark**: {example_name}
- **Variant Name**: {variant_name}
- **Baseline Time**: {baseline_time:.3f}ms
- **Optimized Time**: {patch_time:.3f}ms
- **Speedup**: {actual_speedup:.2f}x
- **GPU**: {environment.gpu_name} ({environment.gpu_arch})
- **CUDA**: {environment.cuda_version}

### Original Code (before optimization)
```python
{original_code[:6000]}
```

### Optimized Code (your best patch)
```python
{patched_code[:6000]}
```

### Instructions

Provide an educational explanation in JSON format:

```json
{{
  "technique_name": "Name of the optimization technique (e.g., 'Memory Coalescing', 'Kernel Fusion', 'Stream Parallelism')",
  "explanation": "A 2-3 sentence plain-language explanation of what changed and why it's faster",
  "why_it_works": "Technical explanation of why this optimization works on this specific GPU architecture ({environment.gpu_arch})",
  "key_concepts": [
    "Concept 1: Brief explanation",
    "Concept 2: Brief explanation",
    "Concept 3: Brief explanation"
  ],
  "performance_impact": {{
    "memory_bandwidth": "How does this affect memory bandwidth utilization?",
    "compute_utilization": "How does this affect GPU compute utilization?",
    "latency": "How does this affect latency?"
  }},
  "when_to_use": "When should developers apply this optimization technique?",
  "when_not_to_use": "When might this optimization be counterproductive?",
  "further_reading": [
    "Topic 1 to learn more about",
    "Topic 2 to learn more about"
  ]
}}
```

Focus on being educational - help the user understand not just WHAT changed, but WHY it's faster and HOW they can apply similar optimizations in their own code.
"""
    
    try:
        response_tuple = analyzer._call_llm(prompt)
        if not response_tuple:
            return None
        
        # _call_llm returns (text, tokens)
        response = response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple
        
        # Extract JSON from response
        import re
        import json
        json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
        if json_match:
            explanation_data = json.loads(json_match.group(1))
            explanation_data['raw_response'] = response
            return explanation_data
        
        # Fallback: return raw response
        return {
            'explanation': response,
            'technique_name': variant_name,
            'raw_response': response,
        }
    except Exception as e:
        logger.warning(f"LLM explanation failed: {e}")
        return None


def test_chapter(
    chapter_dir: Path,
    enable_profiling: bool = False,
    profile_type: str = "none",
    timeout_multiplier: float = 1.0,
    reproducible: bool = False,
    cold_start: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    only_examples: Optional[List[str]] = None,
    accept_regressions: bool = False,
    ncu_metric_set: str = "auto",
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    env_passthrough: Optional[List[str]] = None,
    target_extra_args: Optional[Dict[str, List[str]]] = None,
    # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
    verify_input: bool = True,
    verify_output: bool = True,
    # LLM analysis and patching options
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
) -> Dict[str, Any]:
    return _test_chapter_impl(
        chapter_dir,
        enable_profiling=enable_profiling,
        profile_type=profile_type,
        timeout_multiplier=timeout_multiplier,
        reproducible=reproducible,
        cold_start=cold_start,
        iterations=iterations,
        warmup=warmup,
        only_examples=only_examples,
        accept_regressions=accept_regressions,
        ncu_metric_set=ncu_metric_set,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        env_passthrough=env_passthrough,
        target_extra_args=target_extra_args,
        verify_input=verify_input,
        verify_output=verify_output,
        force_llm=force_llm,
        llm_analysis=llm_analysis,
        llm_provider=llm_provider,
        apply_llm_patches=apply_llm_patches,
        rebenchmark_llm_patches=rebenchmark_llm_patches,
        patch_strategy=patch_strategy,
        llm_patch_retries=llm_patch_retries,
        use_llm_cache=use_llm_cache,
        llm_explain=llm_explain,
    )


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
                    if bench['status'] == 'succeeded':
                        all_speedups.append(bench['best_speedup'])
        
        avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 1.0
        
        f.write("## Overall Summary\n\n")
        f.write(f"- **Chapters tested:** {completed}/{total_chapters}\n")
        f.write(f"- **Chapters skipped:** {skipped} (CUDA unavailable)\n")
        f.write(f"- **Chapters with no benchmarks:** {no_benchmarks}\n")
        total_skipped_hw = sum(r['summary'].get('skipped_hardware', 0) for r in results)
        total_informational = sum(r['summary'].get('informational', 0) for r in results)
        total_regressions = sum(r['summary'].get('failed_regression', 0) for r in results)
        total_failed_errors = sum(r['summary'].get('failed_error', 0) for r in results)
        
        f.write(f"- **Total benchmarks:** {total_benchmarks}\n")
        f.write(f"- **Successful:** {total_successful}\n")
        f.write(f"- **Failed:** {total_failed}\n")
        f.write(f"  - Errors: {total_failed_errors}\n")
        f.write(f"  - Regressions: {total_regressions}\n")
        f.write(f"- **Informational (not benchmarked):** {total_informational}\n")
        if total_skipped_hw > 0:
            f.write(f"- **WARNING: Skipped (hardware/software limitations):** {total_skipped_hw}\n")
        if all_speedups:
            f.write(f"- **Average speedup:** {avg_speedup:.2f}x\n")
            f.write(f"- **Best speedup:** {max(all_speedups):.2f}x\n")
            f.write(f"- **Worst speedup:** {min(all_speedups):.2f}x\n")
        f.write("\n")
        
        # LLM Patch Metrics (if any)
        total_llm_analyzed = sum(r.get('llm_patch_metrics', {}).get('total_analyzed', 0) for r in results)
        if total_llm_analyzed > 0:
            f.write("## LLM Analysis & Patching Summary\n\n")
            total_patches_extracted = sum(r.get('llm_patch_metrics', {}).get('patches_extracted', 0) for r in results)
            total_patches_applied = sum(r.get('llm_patch_metrics', {}).get('patches_applied', 0) for r in results)
            total_patches_failed = sum(r.get('llm_patch_metrics', {}).get('patches_failed', 0) for r in results)
            total_patches_rebenchmarked = sum(r.get('llm_patch_metrics', {}).get('patches_rebenchmarked', 0) for r in results)
            
            f.write(f"- **Benchmarks analyzed:** {total_llm_analyzed}\n")
            f.write(f"- **Patches extracted:** {total_patches_extracted}\n")
            f.write(f"- **Patches applied:** {total_patches_applied}\n")
            f.write(f"- **Patches failed:** {total_patches_failed}\n")
            if total_patches_rebenchmarked > 0:
                f.write(f"- **Patches re-benchmarked:** {total_patches_rebenchmarked}\n")
            
            # Collect all failures
            all_failures = []
            for r in results:
                all_failures.extend(r.get('llm_patch_metrics', {}).get('failures', []))
            
            if all_failures:
                f.write(f"\n### Patch Failures ({len(all_failures)})\n\n")
                f.write("| Example | Failure Reason |\n")
                f.write("|---------|----------------|\n")
                for failure in all_failures[:20]:  # Show first 20
                    reason = failure.get('reason', 'Unknown')[:100]
                    reason = reason.replace('|', '\\|').replace('\n', ' ')
                    f.write(f"| {failure.get('example', 'unknown')} | {reason} |\n")
                if len(all_failures) > 20:
                    f.write(f"\n*...and {len(all_failures) - 20} more failures*\n")
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
                
                bench_status = bench.get('status')
                if bench_status == 'failed_error':
                    f.write(f"- Failed: {bench.get('error', 'Unknown error')}\n")
                elif bench_status == 'failed_regression':
                    f.write(f"- Regression: {bench.get('error', 'Expectation regression detected')}\n")
                elif bench_status == 'skipped':
                    f.write(f"- WARNING: **SKIPPED**: {bench.get('skip_reason', bench.get('error', 'Hardware/software limitation'))}\n")
                else:
                    for opt in bench['optimizations']:
                        if opt['status'] == 'succeeded':
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
        '--targets',
        nargs='+',
        help=("Space-separated list of targets. "
              "Use 'ch3' to test an entire chapter or 'ch3:resnet_50' "
              "to run baseline_resnet_50 and optimized_resnet_50. "
              "Omit this flag (or pass 'all') to run every chapter.")
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
        choices=['none', 'minimal', 'deep_dive', 'roofline'],
        default='none',
        help='Profiling preset: none (default), minimal, deep_dive, or roofline. Non-none enables nsys/ncu/PyTorch profiling.'
    )
    parser.add_argument(
        '--reproducible',
        action='store_true',
        help='Force deterministic seeds/algorithms for reproducible comparisons (slower fallbacks; ops without deterministic support may fail).'
    )
    parser.add_argument(
        '--cold-start',
        action='store_true',
        help='Reset CUDA/GPU state aggressively between benchmarks to emulate cold-start runs.'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Override iteration count for Python benchmarks (default: 20 unless the benchmark defines its own).'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=None,
        help='Override warmup iteration count for Python benchmarks (default: 5 unless the benchmark defines its own).'
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Enable fast smoke-test mode (reduced iterations/warmups, low-memory defaults, expectation checks disabled).'
    )
    parser.add_argument(
        '--launch-via',
        choices=['python', 'torchrun'],
        default='python',
        help='Launcher to use for benchmarks (python or torchrun).'
    )
    parser.add_argument(
        '--nproc-per-node',
        type=int,
        default=None,
        help='torchrun --nproc_per_node value.'
    )
    parser.add_argument(
        '--nnodes',
        type=str,
        default=None,
        help='torchrun --nnodes value.'
    )
    parser.add_argument(
        '--rdzv-backend',
        type=str,
        default=None,
        help='torchrun rendezvous backend (default: c10d when nnodes is set).'
    )
    parser.add_argument(
        '--rdzv-endpoint',
        type=str,
        default=None,
        help='torchrun rendezvous endpoint (host:port).'
    )
    parser.add_argument(
        '--torchrun-env',
        action='append',
        default=None,
        help='Environment variables to forward into torchrun launches (repeatable).'
    )
    parser.add_argument(
        '--target-extra-arg',
        action='append',
        default=None,
        help='Per-target extra args, format: target="--flag value" (repeatable).'
    )
    parser.add_argument(
        '--timeout-multiplier',
        type=float,
        default=1.0,
        help='Multiply all benchmark timeouts by this factor (e.g., 2.0 doubles every timeout).'
    )
    parser.add_argument(
        '--ncu-metric-set',
        choices=['auto', 'minimal', 'deep_dive', 'roofline'],
        default='auto',
        help='Nsight Compute metric preset (auto/minimal/deep_dive/roofline). Auto follows the profile preset.'
    )
    
    args = parser.parse_args()

    # Configure smoke-test mode without leaking env vars.
    set_smoke_mode(bool(args.smoke_test))
    # Refresh benchmark defaults to honor smoke-test toggle.
    set_defaults(BenchmarkDefaults.for_smoke(is_smoke_mode()))

    # In smoke-test validation mode, don't fail the run on expectation drift.
    if is_smoke_mode():
        if hasattr(args, "accept_regressions") and not args.accept_regressions:
            args.accept_regressions = True
            logger.info("Smoke-test mode detected - enabling accept_regressions for quick runs.")
    extra_arg_map: Dict[str, List[str]] = {}
    for entry in args.target_extra_arg or []:
        target, sep, payload = entry.partition("=")
        if not sep or not target or not payload:
            continue
        extra_arg_map[target.strip()] = shlex.split(payload)
    
    logger.info("=" * 80)
    logger.info("TESTING ALL BENCHMARKS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Target override: {args.targets}")

    dump_environment_and_capabilities()
    logger.info("")
    
    # Dump hardware capabilities at start
    logger.info("Dumping hardware capabilities...")
    try:
        dump_caps_path = repo_root / "tools" / "utilities" / "dump_hardware_capabilities.py"
        if dump_caps_path.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(dump_caps_path), "--fast"],
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
    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(args.targets)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"ERROR: {exc}")
        sys.exit(1)
    
    # Test all chapters
    all_results = []
    for chapter_idx, chapter_dir in enumerate(chapter_dirs):
        if not chapter_dir.exists():
            continue

        # GPU reset and cleanup between chapters to prevent state corruption
        if chapter_idx > 0:
            logger.info(f"\n  Resetting GPU state before {chapter_dir.name}...")
            reset_cuda_state()
            reset_gpu_state()
        
        # Clean build directories to prevent stale lock issues
        build_dir = chapter_dir / "build"
        if build_dir.exists():
            try:
                from common.python.build_utils import ensure_clean_build_directory
                ensure_clean_build_directory(build_dir)
            except ImportError:
                pass  # build_utils not available
            except Exception as e:
                logger.warning(f"  Failed to clean build directory: {e}")

        example_filters = chapter_filters.get(chapter_slug(chapter_dir, repo_root))
        only_examples = sorted(example_filters) if example_filters else None
        result = test_chapter(
            chapter_dir,
            enable_profiling=args.profile != 'none',
            profile_type=args.profile,
            timeout_multiplier=args.timeout_multiplier,
            reproducible=args.reproducible,
            cold_start=args.cold_start,
            iterations=args.iterations,
            warmup=args.warmup,
            only_examples=only_examples,
            accept_regressions=args.accept_regressions if hasattr(args, "accept_regressions") else False,
            ncu_metric_set=args.ncu_metric_set,
            launch_via=args.launch_via,
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            rdzv_backend=args.rdzv_backend,
            rdzv_endpoint=args.rdzv_endpoint,
            env_passthrough=args.torchrun_env,
            target_extra_args=extra_arg_map,
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
    total_failed_errors = sum(r['summary'].get('failed_error', 0) for r in all_results)
    total_failed_regressions = sum(r['summary'].get('failed_regression', 0) for r in all_results)
    total_skipped_hw = sum(r['summary'].get('skipped_hardware', 0) for r in all_results)
    total_informational = sum(r['summary'].get('informational', 0) for r in all_results)
    
    logger.info(f"Total benchmarks tested: {total_benchmarks}")
    logger.info(f"Succeeded: {total_successful}")
    logger.info(f"Failed: {total_failed} (errors={total_failed_errors}, regressions={total_failed_regressions})")
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
                if bench['status'] == 'succeeded' and bench['best_speedup'] > 1.0:
                    all_speedups.append(bench['best_speedup'])
    
    if all_speedups:
        logger.info(f"\nSpeedup Statistics:")
        logger.info(f"  Average: {sum(all_speedups)/len(all_speedups):.2f}x")
        logger.info(f"  Best: {max(all_speedups):.2f}x")
        logger.info(f"  Worst: {min(all_speedups):.2f}x")
    
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
