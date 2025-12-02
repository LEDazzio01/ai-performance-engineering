#!/usr/bin/env python3
"""Isolated subprocess runner for benchmarks.

This script receives benchmark configuration via stdin JSON and runs the benchmark
in complete isolation from the parent process. This prevents CUDA context corruption
that can occur when forking after CUDA initialization.

Protocol:
- Input (stdin JSON):
  {
    "benchmark_module_path": "/path/to/benchmark.py",
    "benchmark_class_name": "MyBenchmark" | "get_benchmark",
    "config_dict": {...},
    "device": "cuda:0" | null,
    "initial_state": {...} | null
  }
  
- Output (stdout JSON):
  {
    "success": true/false,
    "result_json": "<serialized PydanticBenchmarkResult>",
    "errors": [...]
  }
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def reset_cuda_state() -> None:
    """Reset CUDA state before benchmark to ensure clean environment."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Reset CUDA graph pool
            if hasattr(torch.cuda, 'graph_pool_trim'):
                try:
                    torch.cuda.graph_pool_trim()
                except Exception:
                    pass
            
            # Reset CUDA RNG state
            try:
                device_idx = torch.cuda.current_device()
                gen = torch.cuda.default_generators[device_idx]
                gen.set_offset(0)
                gen.manual_seed(0)
            except Exception:
                pass
            
            # Reset dynamo/inductor state
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            
            try:
                torch._inductor.cudagraph_trees.reset_cudagraph_trees()
            except Exception:
                pass
    except ImportError:
        pass
    
    gc.collect()


def run_benchmark(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run benchmark and return results in harness-expected format."""
    import importlib.util
    
    errors: List[str] = []
    times_ms: List[float] = []
    memory_peak_mb: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    inference_timing_data: Optional[Dict[str, List[float]]] = None
    
    # Extract input
    module_path = Path(input_data["benchmark_module_path"])
    class_name = input_data["benchmark_class_name"]
    config_dict = input_data.get("config_dict", {})
    device_str = input_data.get("device")
    initial_state = input_data.get("initial_state")
    
    # Config
    iterations = config_dict.get("iterations", 10)
    warmup = config_dict.get("warmup", 3)
    enable_memory_tracking = config_dict.get("enable_memory_tracking", True)
    
    benchmark_name = class_name
    
    try:
        # Reset CUDA state BEFORE loading the module
        reset_cuda_state()
        
        # Load module
        spec = importlib.util.spec_from_file_location("benchmark_module", str(module_path))
        if spec is None or spec.loader is None:
            errors.append(f"Failed to load module spec from {module_path}")
            return _make_error_response(errors)
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = module
        spec.loader.exec_module(module)
        
        # Get benchmark instance
        if class_name == "get_benchmark":
            if not hasattr(module, "get_benchmark"):
                errors.append(f"Module {module_path} has no get_benchmark() function")
                return _make_error_response(errors)
            benchmark = module.get_benchmark()
            benchmark_name = getattr(benchmark, 'name', None) or benchmark.__class__.__name__
        else:
            if not hasattr(module, class_name):
                errors.append(f"Module {module_path} has no class {class_name}")
                return _make_error_response(errors)
            benchmark_class = getattr(module, class_name)
            benchmark = benchmark_class()
            benchmark_name = class_name
        
        # Apply initial state if provided
        if initial_state:
            for key, value in initial_state.items():
                if hasattr(benchmark, key):
                    setattr(benchmark, key, value)
        
        # Setup
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False
        
        if cuda_available:
            import torch
            torch.cuda.reset_peak_memory_stats()
        
        benchmark.setup()
        
        if cuda_available:
            import torch
            torch.cuda.synchronize()
        
        # Warmup
        for _ in range(warmup):
            benchmark.benchmark_fn()
            if cuda_available:
                import torch
                torch.cuda.synchronize()
        
        # Check if benchmark reports its own timing (e.g., CudaBinaryBenchmark parses kernel time from output)
        use_reported_time = bool(getattr(benchmark, "use_reported_time", False))
        
        # Timed runs - use CUDA events for GPU (accurate), perf_counter for CPU only
        if cuda_available:
            import torch
            # GPU: use CUDA Events for accurate GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()  # sync once before timing loop
            
            for _ in range(iterations):
                start_event.record()
                benchmark.benchmark_fn()
                end_event.record()
                end_event.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                
                # Use benchmark-reported time if available (e.g., parsed from CUDA binary output)
                if use_reported_time:
                    reported = getattr(benchmark, "last_time_ms", None)
                    if reported is not None:
                        elapsed_ms = reported
                
                times_ms.append(elapsed_ms)
        else:
            # CPU: use perf_counter (only valid for CPU-only benchmarks)
            for _ in range(iterations):
                start = time.perf_counter()
                benchmark.benchmark_fn()
                elapsed_ms = (time.perf_counter() - start) * 1000
                times_ms.append(elapsed_ms)
        
        # Memory tracking
        if enable_memory_tracking and cuda_available:
            import torch
            memory_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            memory_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Teardown
        benchmark.teardown()
        
        # Get inference timing data if available
        if hasattr(benchmark, "get_inference_timing_data"):
            inference_timing_data = benchmark.get_inference_timing_data()
        
    except Exception as e:
        tb = traceback.format_exc()
        errors.append(f"Benchmark execution failed: {e}")
        errors.append(tb)
        return _make_error_response(errors)
    
    # Build successful response with Pydantic model
    return _make_success_response(
        times_ms=times_ms,
        iterations=iterations,
        warmup=warmup,
        memory_peak_mb=memory_peak_mb,
        memory_allocated_mb=memory_allocated_mb,
        benchmark_name=benchmark_name,
        device_str=device_str,
        inference_timing_data=inference_timing_data,
        errors=errors,
    )


def _make_error_response(errors: List[str]) -> Dict[str, Any]:
    """Create error response in harness-expected format."""
    # Build a minimal BenchmarkResult with errors
    from core.benchmark.models import BenchmarkResult, TimingStats
    
    result = BenchmarkResult(
        timing=TimingStats(
            mean_ms=0.0,
            median_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            iterations=0,
            warmup_iterations=0,
            raw_times_ms=[],
        ),
        errors=errors,
    )
    
    return {
        "success": False,
        "result_json": result.model_dump_json(),
        "errors": errors,
    }


def _make_success_response(
    times_ms: List[float],
    iterations: int,
    warmup: int,
    memory_peak_mb: Optional[float],
    memory_allocated_mb: Optional[float],
    benchmark_name: str,
    device_str: Optional[str],
    inference_timing_data: Optional[Dict[str, List[float]]],
    errors: List[str],
) -> Dict[str, Any]:
    """Create success response in harness-expected format."""
    from core.benchmark.models import BenchmarkResult, TimingStats, MemoryStats, InferenceTimingStats
    
    # Calculate timing statistics
    if times_ms:
        mean_ms = statistics.mean(times_ms)
        median_ms = statistics.median(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        min_ms = min(times_ms)
        max_ms = max(times_ms)
    else:
        mean_ms = median_ms = std_ms = min_ms = max_ms = 0.0
    
    # Build timing stats
    timing = TimingStats(
        mean_ms=mean_ms,
        median_ms=median_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        iterations=iterations,
        warmup_iterations=warmup,
        raw_times_ms=times_ms,
    )
    
    # Build memory stats
    memory = None
    if memory_peak_mb is not None:
        memory = MemoryStats(
            peak_mb=memory_peak_mb,
            allocated_mb=memory_allocated_mb,
        )
    
    # Build inference timing stats
    inference_timing = None
    if inference_timing_data:
        inference_timing = InferenceTimingStats(**inference_timing_data)
    
    # Build full result
    result = BenchmarkResult(
        timing=timing,
        memory=memory,
        inference_timing=inference_timing,
        benchmark_name=benchmark_name,
        device=device_str,
        errors=errors,
    )
    
    return {
        "success": True,
        "result_json": result.model_dump_json(),
        "errors": errors,
    }


def main() -> None:
    """Main entry point - read JSON from stdin, run benchmark, write JSON to stdout."""
    try:
        # Read input JSON from stdin
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        # Run benchmark
        result = run_benchmark(input_data)
        
        # Write result JSON to stdout
        print(json.dumps(result))
        
    except json.JSONDecodeError as e:
        error_result = _make_error_response([f"Failed to parse input JSON: {e}"])
        print(json.dumps(error_result))
        sys.exit(1)
    except Exception as e:
        error_result = _make_error_response([f"Runner failed: {e}", traceback.format_exc()])
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
