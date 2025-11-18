"""Isolated benchmark runner for subprocess-based execution.

Provides robust process isolation for benchmarks with reliable timeout cancellation.
Benchmarks are executed in a child process that can be killed if it exceeds the timeout.
Uses Pydantic models for type-safe result serialization.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

# Ensure repo root is importable even when this file is executed via an absolute path
_REPO_ROOT: Optional[Path] = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "common" / "__init__.py"
    if candidate.exists():
        _REPO_ROOT = parent
        break
if _REPO_ROOT is None:
    # Fallback to the directory two levels up (common/python -> repo root)
    try:
        _REPO_ROOT = Path(__file__).resolve().parents[2]
    except IndexError:
        _REPO_ROOT = Path(__file__).resolve().parent
repo_root_str = str(_REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import torch

if TYPE_CHECKING:
    from common.python.benchmark_models import BenchmarkResult, MemoryStats

# Pydantic is required - fail fast if not available
from common.python.benchmark_models import BenchmarkResult, MemoryStats

PYDANTIC_AVAILABLE = True

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkConfig, BenchmarkMode, ExecutionMode


def run_benchmark_isolated(
    benchmark_module_path: str,
    benchmark_class_name: str,
    config_dict: Dict[str, Any],
    device: Optional[str] = None,
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
        
    Returns:
        Dictionary with keys: success, result_json (serialized BenchmarkResult), errors
    """
    result: Dict[str, Any] = {
        "success": False,
        "result_json": None,
        "errors": [],
    }
    
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
        benchmark_result: BenchmarkResult = harness.benchmark(benchmark)
        
        # Serialize Pydantic model to JSON
        result["success"] = True
        result["result_json"] = benchmark_result.model_dump_json()
        
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        # Specific errors for benchmark discovery/loading
        result["errors"].append(f"Benchmark discovery failed: {type(e).__name__}: {str(e)}")
        result["errors"].append(f"Traceback: {traceback.format_exc()}")
    except (subprocess.TimeoutExpired, TimeoutError) as e:
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
    
    Outputs JSON result to stdout with:
    - success: bool
    - result_json: Serialized BenchmarkResult (Pydantic JSON)
    - errors: List of error messages
    """
    try:
        input_data = json.loads(sys.stdin.read())
        benchmark_module_path = input_data["benchmark_module_path"]
        benchmark_class_name = input_data["benchmark_class_name"]
        config_dict = input_data.get("config_dict", {})
        device = input_data.get("device")
        
        result = run_benchmark_isolated(
            benchmark_module_path,
            benchmark_class_name,
            config_dict,
            device
        )
        
        # Output result as JSON
        print(json.dumps(result))
        
    except (json.JSONDecodeError, KeyError) as e:
        # Input parsing errors
        error_result = {
            "success": False,
            "errors": [f"Invalid input: {type(e).__name__}: {str(e)}", traceback.format_exc()],
        }
        print(json.dumps(error_result))
        sys.exit(1)
    except Exception as e:
        # Catch-all for unexpected errors
        error_result = {
            "success": False,
            "errors": [f"Subprocess execution failed: {type(e).__name__}: {str(e)}", traceback.format_exc()],
            "result_json": None,
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
