#!/usr/bin/env python3
"""Isolated benchmark runner for subprocess-based execution.

Provides robust process isolation for benchmarks with reliable timeout cancellation.
Benchmarks are executed in a child process that can be killed if it exceeds the timeout.
Uses Pydantic models for type-safe result serialization.

IMPORTANT: Heavy imports (torch, benchmark_harness, etc.) are DEFERRED to runtime
to ensure the script can always output valid JSON errors even if imports fail.
"""

from __future__ import annotations

# ONLY import stdlib at module level - everything else is deferred
import json
import os
import sys
import traceback
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from common.python.benchmark_models import BenchmarkResult, MemoryStats


def _setup_repo_path() -> str:
    """Ensure repo root is importable even when this file is executed via an absolute path."""
    repo_root: Optional[Path] = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "common" / "__init__.py"
        if candidate.exists():
            repo_root = parent
            break
    if repo_root is None:
        # Fallback to the directory two levels up (common/python -> repo root)
        try:
            repo_root = Path(__file__).resolve().parents[2]
        except IndexError:
            repo_root = Path(__file__).resolve().parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root_str


def run_benchmark_isolated(
    benchmark_module_path: str,
    benchmark_class_name: str,
    config_dict: Dict[str, Any],
    device: Optional[str] = None,
    initial_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a benchmark in the current process (called from subprocess).
    
    This function is designed to be called from a subprocess spawned by
    the benchmark harness. It imports the benchmark module, instantiates
    the benchmark, runs it, and returns results as a JSON-serializable dictionary.
    
    Args:
        benchmark_module_path: Path to the benchmark module file
        benchmark_class_name: Name of the benchmark class or 'get_benchmark' function
        config_dict: Dictionary of BenchmarkConfig values to override
        device: CUDA device string (e.g., 'cuda:0') or None for auto-detect
        initial_state: Optional mapping of simple attribute values to set before setup
        
    Returns:
        Dictionary with keys: success, result_json (serialized BenchmarkResult), errors
    """
    result: Dict[str, Any] = {
        "success": False,
        "result_json": None,
        "errors": [],
    }
    
    # DEFERRED IMPORTS - done here so any failures are caught and reported as JSON
    try:
        import torch
    except ImportError as e:
        result["errors"].append(f"Failed to import torch: {e}")
        result["errors"].append(traceback.format_exc())
        return result
    
    try:
        from common.python.benchmark_models import BenchmarkResult, MemoryStats
        from common.python.benchmark_harness import (
            BaseBenchmark, BenchmarkHarness, BenchmarkConfig, 
            BenchmarkMode, ExecutionMode
        )
    except ImportError as e:
        result["errors"].append(f"Failed to import benchmark harness: {e}")
        result["errors"].append(traceback.format_exc())
        return result
    
    try:
        # Add repo root to path so we can import common.python modules
        module_path = Path(benchmark_module_path).resolve()
        repo_root = module_path
        while repo_root.parent != repo_root:  # Not at filesystem root
            if (repo_root / "common" / "python").exists():
                break
            repo_root = repo_root.parent
        else:
            # Fallback: assume common/python is sibling to benchmark module
            repo_root = module_path.parent.parent.parent
        
        sys.path.insert(0, str(repo_root))
        
        if not module_path.exists():
            result["errors"].append(f"Benchmark module not found: {benchmark_module_path}")
            return result
        
        module_dir = module_path.parent
        sys.path.insert(0, str(module_dir))
        
        # Import the benchmark module
        module_name = module_path.stem
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            result["errors"].append(f"Could not load module spec from {module_path}")
            return result
        
        benchmark_module = importlib.util.module_from_spec(spec)
        # Ensure module is visible to dataclasses/typing lookups during execution.
        sys.modules[module_name] = benchmark_module
        spec.loader.exec_module(benchmark_module)
        
        # Get benchmark instance
        benchmark = None
        if hasattr(benchmark_module, "get_benchmark"):
            benchmark = benchmark_module.get_benchmark()
        elif hasattr(benchmark_module, benchmark_class_name):
            benchmark_class = getattr(benchmark_module, benchmark_class_name)
            benchmark = benchmark_class()
        else:
            # Try to find any class with benchmark_fn method
            for attr_name in dir(benchmark_module):
                attr = getattr(benchmark_module, attr_name)
                if isinstance(attr, type) and hasattr(attr, "benchmark_fn") and callable(getattr(attr, "benchmark_fn", None)):
                    benchmark = attr()
                    break
        
        if benchmark is None:
            result["errors"].append(f"Could not find or instantiate benchmark: {benchmark_class_name}")
            return result
        
        # Create config from dict
        config = BenchmarkConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                # For percentiles, ensure it's always a list (never None)
                if key == 'percentiles':
                    setattr(config, key, value if value is not None and isinstance(value, list) else [25, 50, 75, 99])
                elif value is not None:
                    setattr(config, key, value)
        
        # CRITICAL: Ensure percentiles is set even if not in config_dict
        # This handles cases where percentiles wasn't serialized
        if config.percentiles is None:
            config.percentiles = [25, 50, 75, 99]

        if initial_state:
            for key, value in initial_state.items():
                try:
                    setattr(benchmark, key, value)
                except Exception:
                    # Best-effort propagation; ignore attributes that cannot be set.
                    pass
        
        # Disable subprocess in runner to avoid recursion (runner IS the subprocess)
        config.use_subprocess = False
        config.execution_mode = ExecutionMode.THREAD
        config._sync_launch_via()
        config._sync_execution_mode()
        
        # Create harness
        harness = BenchmarkHarness(
            mode=BenchmarkMode.CUSTOM,
            config=config
        )
        
        # Override device if specified
        if device:
            harness.device = torch.device(device)
        
        # Run benchmark - returns Pydantic BenchmarkResult
        benchmark_result = harness.benchmark(benchmark)
        
        # Serialize Pydantic model to JSON
        result["success"] = True
        result["result_json"] = benchmark_result.model_dump_json()
        
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        # Specific errors for benchmark discovery/loading
        result["errors"].append(f"Benchmark discovery failed: {type(e).__name__}: {str(e)}")
        result["errors"].append(f"Traceback: {traceback.format_exc()}")
    except TimeoutError as e:
        # Timeout errors
        result["errors"].append(f"Benchmark execution timed out: {str(e)}")
        result["errors"].append(f"Traceback: {traceback.format_exc()}")
    except RuntimeError as e:
        # Runtime errors (CUDA, device, etc.)
        result["errors"].append(f"Benchmark execution failed: {type(e).__name__}: {str(e)}")
        result["errors"].append(f"Traceback: {traceback.format_exc()}")
    except Exception as e:
        # Catch-all for unexpected errors
        result["errors"].append(f"Unexpected error during benchmark execution: {type(e).__name__}: {str(e)}")
        result["errors"].append(f"Traceback: {traceback.format_exc()}")
    
    return result


def main():
    """Entry point for subprocess execution.
    
    Expects JSON input on stdin with:
    - benchmark_module_path: Path to benchmark module
    - benchmark_class_name: Name of benchmark class
    - config_dict: BenchmarkConfig overrides
    - device: Optional device string
    - initial_state: Optional mapping of simple attribute values

    Outputs JSON result to stdout with:
    - success: bool
    - result_json: Serialized BenchmarkResult (Pydantic JSON)
    - errors: List of error messages
    """
    # Setup repo path first
    _setup_repo_path()
    
    # Read and parse input
    raw_input = sys.stdin.read()
    try:
        input_data = json.loads(raw_input)
        benchmark_module_path = input_data["benchmark_module_path"]
        benchmark_class_name = input_data["benchmark_class_name"]
        config_dict = input_data.get("config_dict", {})
        device = input_data.get("device")
        initial_state = input_data.get("initial_state")
    except Exception as e:
        # Input parsing errors
        error_result = {
            "success": False,
            "result_json": None,
            "errors": [
                f"Invalid input: {type(e).__name__}: {str(e)}",
                traceback.format_exc(),
                f"Raw input: {raw_input[:1000] if len(raw_input) > 1000 else raw_input}",
            ],
        }
        print(json.dumps(error_result))
        sys.exit(1)

    stdout_buffer = StringIO()
    try:
        with redirect_stdout(stdout_buffer):
            result = run_benchmark_isolated(
                benchmark_module_path,
                benchmark_class_name,
                config_dict,
                device,
                initial_state,
            )
    except Exception as e:
        # Catch-all for unexpected errors (ensure JSON-only stdout)
        error_result = {
            "success": False,
            "result_json": None,
            "errors": [
                f"Subprocess execution failed: {type(e).__name__}: {str(e)}",
                traceback.format_exc(),
            ],
        }
        captured = stdout_buffer.getvalue()
        if captured:
            error_result["stdout"] = captured
        print(json.dumps(error_result))
        sys.exit(1)

    captured_stdout = stdout_buffer.getvalue()
    if captured_stdout:
        # Surface captured stdout for debugging; parent harness ignores this key.
        result["stdout"] = captured_stdout

    # Output result as JSON (only)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
