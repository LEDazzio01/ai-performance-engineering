"""Production-grade benchmarking harness with profiling integration.

Provides industry-standard benchmarking using Triton do_bench, PyTorch Timer,
and custom CUDA Events. Supports nsys, ncu, and PyTorch profiler integration.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import torch


class BenchmarkMode(Enum):
    """Benchmarking mode selection."""
    TRITON = "triton"  # Use triton.testing.do_bench
    PYTORCH = "pytorch"  # Use torch.utils.benchmark.Timer
    CUSTOM = "custom"  # Use CUDA Events / time.perf_counter


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    iterations: int = 100
    warmup: int = 10
    min_run_time_ms: float = 100.0  # Minimum total runtime for PyTorch Timer
    percentiles: List[float] = field(default_factory=lambda: [25, 50, 75, 99])
    enable_memory_tracking: bool = False
    deterministic: bool = False
    seed: Optional[int] = None
    device: Optional[torch.device] = None
    enable_profiling: bool = True  # Enable nsys/ncu/PyTorch profiler (default: True - always profiling)
    enable_nsys: bool = False  # Enable nsys profiling (requires CUDA, wraps entire process)
    enable_ncu: bool = False  # Enable ncu profiling (requires CUDA, wraps entire process)
    profiling_output_dir: Optional[str] = None  # Directory for profiling outputs
    enable_nvtx: Optional[bool] = None  # Enable NVTX ranges (None = auto: True if profiling, False otherwise)
    timeout_seconds: int = 15  # Required timeout for benchmark execution in seconds (prevents hangs) - DEFAULT 15s
    # Note: Setup/teardown (including compilation) are not subject to timeout,
    # but should complete within reasonable time or fail with error
    # nsys/ncu profiling timeout is separate and typically longer (60-120s)
    nsys_timeout_seconds: int = 120  # Timeout for nsys profiling runs
    ncu_timeout_seconds: int = 180  # Timeout for ncu profiling runs (ncu can be slow)
    
    def __post_init__(self):
        """Set enable_nvtx based on profiling if not explicitly set."""
        if self.enable_nvtx is None:
            # Auto-enable NVTX when profiling is enabled (for nsys traces)
            # Since enable_profiling=True by default, NVTX will be enabled by default
            self.enable_nvtx = self.enable_profiling


@dataclass
class BenchmarkResult:
    """Statistical results from benchmarking."""
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    percentiles: Dict[float, float]  # e.g., {25.0: 1.23, 50.0: 1.45, ...}
    iterations: int
    warmup_iterations: int
    memory_peak_mb: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    profiling_outputs: Dict[str, str] = field(default_factory=dict)  # Paths to profiling files
    nsys_metrics: Dict[str, float] = field(default_factory=dict)  # Extracted nsys metrics
    ncu_metrics: Dict[str, float] = field(default_factory=dict)  # Extracted ncu metrics


class Benchmark(Protocol):
    """Protocol for benchmarkable implementations."""
    
    def setup(self) -> None:
        """Setup phase: initialize models, data, etc."""
        ...
    
    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args."""
        ...
    
    def teardown(self) -> None:
        """Cleanup phase."""
        ...
    
    def get_config(self) -> Optional[BenchmarkConfig]:
        """Optional: return benchmark-specific config overrides."""
        return None
    
    def validate_result(self) -> Optional[str]:
        """Optional: validate benchmark result, return error message if invalid."""
        return None


class BenchmarkHarness:
    """Production-grade benchmarking harness with profiling support."""
    
    def __init__(
        self,
        mode: BenchmarkMode = BenchmarkMode.CUSTOM,
        config: Optional[BenchmarkConfig] = None
    ):
        self.mode = mode
        self.config = config or BenchmarkConfig()
        self.device = self.config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._setup_reproducibility()
    
    def _setup_reproducibility(self) -> None:
        """Setup for reproducible benchmarks."""
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
    
    @contextmanager
    def _memory_tracking(self):
        """Context manager for memory tracking."""
        if not self.config.enable_memory_tracking or not torch.cuda.is_available():
            yield None
            return
        
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        yield
        torch.cuda.synchronize(self.device)
        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
        # Return values via generator protocol - caller gets tuple
        return (peak_mb, allocated_mb)
    
    def benchmark(self, benchmark: Benchmark) -> BenchmarkResult:
        """Run benchmark and return statistical results.
        
        Uses threading timeout (required) to prevent hangs. Default timeout is 15 seconds.
        """
        # Clone config to avoid mutating shared instance
        from dataclasses import replace
        config = replace(self.config)
        bench_config = benchmark.get_config()
        if bench_config:
            # Override with benchmark-specific settings
            for key, value in bench_config.__dict__.items():
                if value is not None:
                    setattr(config, key, value)
        
        errors = []
        memory_peak_mb = None
        memory_allocated_mb = None
        profiling_outputs = {}
        nsys_metrics = {}
        ncu_metrics = {}
        times_ms = []
        
        def run_benchmark_internal():
            """Internal benchmark execution function."""
            nonlocal times_ms, memory_peak_mb, memory_allocated_mb, profiling_outputs, errors, nsys_metrics, ncu_metrics
            
            try:
                # Setup - this may include CUDA extension compilation OR torch.compile()
                # IMPORTANT: Setup MUST complete quickly or timeout will occur
                # torch.compile() compilation can hang - timeout will catch it
                # If setup takes longer than timeout, it will be killed by the outer timeout
                import signal
                import time
                start_time = time.time()
                benchmark.setup()
                setup_time = time.time() - start_time
                if setup_time > config.timeout_seconds * 0.8:  # Warn if setup takes >80% of timeout
                    print(f"  WARNING: Setup took {setup_time:.1f}s (near timeout limit)")
                
                # Warmup
                self._warmup(benchmark.benchmark_fn, config.warmup)
                
                # Memory tracking: Reset stats BEFORE benchmark to capture peak during execution
                if config.enable_memory_tracking and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.device)
                    torch.cuda.synchronize(self.device)
                
                # Benchmark using selected mode
                # Note: nsys/ncu profiling wraps the entire process, so it's handled separately
                if config.enable_nsys or config.enable_ncu:
                    # Run nsys/ncu profiling (these wrap the entire process)
                    result = self._benchmark_with_nsys_ncu(benchmark, config)
                    times_ms = result.get("times_ms", [])
                    profiling_outputs = result.get("profiling_outputs", {})
                    nsys_metrics = result.get("nsys_metrics", {})
                    ncu_metrics = result.get("ncu_metrics", {})
                elif config.enable_profiling:
                    times_ms, profiling_outputs = self._benchmark_with_profiling(
                        benchmark.benchmark_fn, config
                    )
                else:
                    times_ms = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
                
                # Memory tracking: Get stats AFTER benchmark to capture peak during execution
                if config.enable_memory_tracking and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                    memory_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                    memory_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
                
                # Validate result
                validation_error = benchmark.validate_result()
                if validation_error:
                    errors.append(f"Validation failed: {validation_error}")
                
            except Exception as e:
                errors.append(f"Benchmark execution failed: {str(e)}")
                times_ms = []
            finally:
                # Always cleanup
                try:
                    benchmark.teardown()
                except Exception as e:
                    errors.append(f"Teardown failed: {str(e)}")
                
                # Force cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # ALWAYS run with timeout (required, default 15 seconds)
        execution_result = {"done": False, "error": None}
        
        def run_with_result():
            try:
                run_benchmark_internal()
            except Exception as e:
                execution_result["error"] = e
            finally:
                execution_result["done"] = True
        
        # Only print timeout message if timeout actually occurs (not upfront)
        thread = threading.Thread(target=run_with_result, daemon=True)
        thread.start()
        thread.join(timeout=config.timeout_seconds)
        
        if not execution_result["done"]:
            # TIMEOUT OCCURRED - make it very clear
            print("\n" + "=" * 80)
            print("TIMEOUT: Benchmark execution exceeded timeout limit")
            print("=" * 80)
            print(f"   Timeout limit: {config.timeout_seconds} seconds")
            print(f"   Status: Benchmark did not complete within timeout period")
            print(f"   Action: Benchmark execution was terminated to prevent hang")
            print("=" * 80)
            print()
            
            errors.append(f"TIMEOUT: Benchmark exceeded timeout of {config.timeout_seconds} seconds")
            times_ms = []
            # Aggressive cleanup on timeout - CUDA operations can hang
            try:
                benchmark.teardown()
            except:
                pass
            # Force CUDA cleanup - critical after timeout
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()  # Clean up IPC resources
                    torch.cuda.reset_peak_memory_stats()  # Reset stats
                except:
                    pass
            gc.collect()
            # Force another GC pass to clean up any remaining references
            gc.collect()
        elif execution_result["error"]:
            errors.append(f"Benchmark execution error: {str(execution_result['error'])}")
            times_ms = []
        # Don't print success message for normal completion - only print on timeout/failure
        
        if not times_ms:
            raise RuntimeError(f"Benchmark failed: {', '.join(errors)}")
        
        # Compute statistics
        result = self._compute_stats(times_ms, config)
        result.memory_peak_mb = memory_peak_mb
        result.memory_allocated_mb = memory_allocated_mb
        result.errors = errors
        result.profiling_outputs = profiling_outputs
        result.nsys_metrics = nsys_metrics
        result.ncu_metrics = ncu_metrics
        
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
            
            # Run benchmark with PyTorch profiler
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
                # Run benchmark iterations with minimal overhead
                times_ms = []
                is_cuda = self.device.type == "cuda"
                
                if is_cuda:
                    # Create events once, reuse across iterations
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(self.device)  # Sync once before loop
                    
                    for _ in range(config.iterations):
                        start_event.record()
                        fn()
                        end_event.record()
                        torch.cuda.synchronize(self.device)
                        times_ms.append(start_event.elapsed_time(end_event))
                        prof.step()  # Record each iteration in profiling trace
                else:
                    # CPU: use high-resolution timer
                    for _ in range(config.iterations):
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
            return self._benchmark_without_profiling(fn, config), {}
    
    def _benchmark_with_nsys_ncu(
        self, benchmark: Benchmark, config: BenchmarkConfig
    ) -> Dict[str, Any]:
        """Benchmark with nsys/ncu profiling enabled.
        
        Note: nsys and ncu wrap the entire Python process, so we need to create
        a wrapper script that imports and runs the benchmark.
        """
        if not torch.cuda.is_available():
            # Fallback to regular profiling if CUDA not available
            times_ms, profiling_outputs = self._benchmark_with_profiling(
                benchmark.benchmark_fn, config
            )
            return {
                "times_ms": times_ms,
                "profiling_outputs": profiling_outputs,
                "nsys_metrics": {},
                "ncu_metrics": {},
            }
        
        # Create profiling output directory
        if config.profiling_output_dir:
            prof_dir = Path(config.profiling_output_dir)
            prof_dir.mkdir(parents=True, exist_ok=True)
        else:
            prof_dir = Path("profiling_results")
            prof_dir.mkdir(parents=True, exist_ok=True)
        
        times_ms = []
        profiling_outputs = {}
        nsys_metrics = {}
        ncu_metrics = {}
        
        # Get benchmark module and class info for wrapper script
        benchmark_module = inspect.getmodule(benchmark)
        benchmark_class = benchmark.__class__.__name__
        
        # Run regular benchmark first to get timing (nsys/ncu are for metrics, not timing)
        times_ms = self._benchmark_without_profiling(benchmark.benchmark_fn, config)
        
        # Run nsys profiling if enabled
        if config.enable_nsys:
            nsys_result = self._run_nsys_profiling(benchmark, benchmark_module, benchmark_class, prof_dir, config)
            if nsys_result:
                profiling_outputs.update(nsys_result.get("profiling_outputs", {}))
                nsys_metrics = nsys_result.get("metrics", {})
        
        # Run ncu profiling if enabled
        if config.enable_ncu:
            ncu_result = self._run_ncu_profiling(benchmark, benchmark_module, benchmark_class, prof_dir, config)
            if ncu_result:
                profiling_outputs.update(ncu_result.get("profiling_outputs", {}))
                ncu_metrics = ncu_result.get("metrics", {})
        
        return {
            "times_ms": times_ms,
            "profiling_outputs": profiling_outputs,
            "nsys_metrics": nsys_metrics,
            "ncu_metrics": ncu_metrics,
        }
    
    def _run_nsys_profiling(
        self, benchmark: Benchmark, benchmark_module, benchmark_class: str,
        prof_dir: Path, config: BenchmarkConfig
    ) -> Optional[Dict[str, Any]]:
        """Run nsys profiling on benchmark."""
        try:
            # Check if nsys is available
            subprocess.run(["nsys", "--version"], capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        
        # Create wrapper script
        wrapper_script = self._create_benchmark_wrapper(
            benchmark, benchmark_module, benchmark_class, config
        )
        
        if not wrapper_script:
            return None
        
        try:
            # Create output path
            nsys_output = prof_dir / f"nsys_{benchmark_class}"
            
            # Build nsys command
            nsys_command = [
                "nsys",
                "profile",
                "--force-overwrite=true",
                "-o",
                str(nsys_output),
                "-t", "cuda,nvtx,osrt,cublas,cudnn",
                "-s", "cpu",
                "--python-sampling=true",
                "--python-sampling-frequency=1000",
                "--cudabacktrace=true",
                "--stats=true",
                sys.executable,
                str(wrapper_script)
            ]
            
            # Run nsys
            result = subprocess.run(
                nsys_command,
                capture_output=True,
                timeout=config.nsys_timeout_seconds,
                check=False
            )
            
            # Find the generated .nsys-rep file
            nsys_rep_path = Path(f"{nsys_output}.nsys-rep")
            if not nsys_rep_path.exists():
                # Try alternative naming
                for rep_file in prof_dir.glob(f"nsys_{benchmark_class}*.nsys-rep"):
                    nsys_rep_path = rep_file
                    break
            
            if nsys_rep_path.exists():
                profiling_outputs = {"nsys_rep": str(nsys_rep_path)}
                # Extract metrics
                metrics = self._extract_nsys_metrics(nsys_rep_path)
                return {
                    "profiling_outputs": profiling_outputs,
                    "metrics": metrics,
                }
        except Exception as e:
            pass
        finally:
            # Clean up wrapper script
            try:
                if wrapper_script.exists():
                    wrapper_script.unlink()
            except:
                pass
        
        return None
    
    def _run_ncu_profiling(
        self, benchmark: Benchmark, benchmark_module, benchmark_class: str,
        prof_dir: Path, config: BenchmarkConfig
    ) -> Optional[Dict[str, Any]]:
        """Run ncu profiling on benchmark."""
        try:
            # Check if ncu is available
            subprocess.run(["ncu", "--version"], capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        
        # Create wrapper script
        wrapper_script = self._create_benchmark_wrapper(
            benchmark, benchmark_module, benchmark_class, config
        )
        
        if not wrapper_script:
            return None
        
        try:
            # Create output path
            ncu_output = prof_dir / f"ncu_{benchmark_class}"
            
            # Build ncu command
            ncu_command = [
                "ncu",
                "--set", "full",
                "--metrics", "gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active",
                "--replay-mode", "kernel",
                "-o", str(ncu_output),
                sys.executable,
                str(wrapper_script)
            ]
            
            # Run ncu
            result = subprocess.run(
                ncu_command,
                capture_output=True,
                timeout=config.ncu_timeout_seconds,
                check=False
            )
            
            # Find the generated .ncu-rep file
            ncu_rep_path = Path(f"{ncu_output}.ncu-rep")
            if not ncu_rep_path.exists():
                # Try alternative naming
                for rep_file in prof_dir.glob(f"ncu_{benchmark_class}*.ncu-rep"):
                    ncu_rep_path = rep_file
                    break
            
            if ncu_rep_path.exists():
                profiling_outputs = {"ncu_rep": str(ncu_rep_path)}
                # Extract metrics
                metrics = self._extract_ncu_metrics(ncu_rep_path)
                return {
                    "profiling_outputs": profiling_outputs,
                    "metrics": metrics,
                }
        except Exception as e:
            pass
        finally:
            # Clean up wrapper script
            try:
                if wrapper_script.exists():
                    wrapper_script.unlink()
            except:
                pass
        
        return None
    
    def _create_benchmark_wrapper(
        self, benchmark: Benchmark, benchmark_module, benchmark_class: str, config: BenchmarkConfig
    ) -> Optional[Path]:
        """Create a temporary Python script that runs the benchmark.
        
        The wrapper script imports the benchmark module and recreates the benchmark
        instance, then runs setup, warmup, and profiling iterations.
        """
        try:
            # Get module path
            if benchmark_module is None:
                return None
            
            module_name = benchmark_module.__name__
            module_file = getattr(benchmark_module, "__file__", None)
            
            if module_file is None:
                return None
            
            module_path = Path(module_file).resolve()
            module_dir = module_path.parent
            
            # Create temporary wrapper script
            wrapper_script = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir=tempfile.gettempdir()
            )
            
            # Determine how to instantiate the benchmark
            # Try common patterns: get_benchmark() function, or class name
            instantiation_code = f"""# Get benchmark instance (try common patterns)
benchmark = None
try:
    if hasattr({module_name}, "get_benchmark"):
        benchmark = {module_name}.get_benchmark()
    elif hasattr({module_name}, "{benchmark_class}"):
        benchmark_class = getattr({module_name}, "{benchmark_class}")
        benchmark = benchmark_class()
    else:
        # Try to find any class with benchmark_fn method
        for attr_name in dir({module_name}):
            attr = getattr({module_name}, attr_name)
            if isinstance(attr, type) and hasattr(attr, "benchmark_fn") and callable(getattr(attr, "benchmark_fn", None)):
                benchmark = attr()
                break
except Exception as e:
    import traceback
    print("Error creating benchmark: " + str(e))
    traceback.print_exc()
    raise

if benchmark is None:
    raise RuntimeError("Could not find or instantiate benchmark instance")
"""
            
            wrapper_content = f'''import sys
from pathlib import Path

# Add module directory to path
sys.path.insert(0, r"{module_dir}")

# Import the benchmark module
import {module_name}

{instantiation_code}

# Run benchmark
try:
    benchmark.setup()
    
    # Warmup
    for _ in range({config.warmup}):
        benchmark.benchmark_fn()
    
    # Synchronize before profiling
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Run benchmark iterations for profiling (limited for profiling overhead)
    for _ in range({min(config.iterations, 10)}):
        benchmark.benchmark_fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    benchmark.teardown()
except Exception as e:
    import traceback
    print("Error running benchmark: " + str(e))
    traceback.print_exc()
    raise
'''
            
            wrapper_script.write(wrapper_content)
            wrapper_script.close()
            
            return Path(wrapper_script.name)
        except Exception as e:
            return None
    
    def _extract_nsys_metrics(self, nsys_rep_path: Path) -> Dict[str, float]:
        """Extract metrics from nsys report file."""
        metrics = {}
        
        if not nsys_rep_path.exists():
            return metrics
        
        # Try using nsys stats command
        try:
            result = subprocess.run(
                ["nsys", "stats", "--report", "cuda_gpu_sum", "--format", "csv", str(nsys_rep_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                csv_metrics = self._parse_nsys_csv(result.stdout)
                metrics.update(csv_metrics)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        # Also try using the extract_nsys_summary module if available
        try:
            # Try to import and use the existing extraction tool
            import sys
            repo_root = Path(__file__).parent.parent.parent.parent
            tools_path = repo_root / "tools" / "profiling"
            if str(tools_path) not in sys.path:
                sys.path.insert(0, str(tools_path))
            
            from extract_nsys_summary import harvest
            harvested = harvest(nsys_rep_path)
            
            # Convert harvested metrics to dict format
            for entry in harvested:
                metric_name = entry.get("metric", "")
                value_str = entry.get("value", "")
                if metric_name and value_str:
                    try:
                        # Try to parse as float
                        value = float(value_str.replace(",", "").replace("%", ""))
                        # Create a clean metric name
                        clean_name = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                        metrics[f"nsys_{clean_name}"] = value
                    except (ValueError, AttributeError):
                        pass
        except (ImportError, Exception):
            pass
        
        return metrics
    
    def _parse_nsys_csv(self, csv_text: str) -> Dict[str, float]:
        """Parse nsys CSV output for timing metrics."""
        metrics = {}
        
        # Extract total GPU time
        match = re.search(r"Total GPU Time.*?,(\d+\.?\d*)", csv_text, re.IGNORECASE)
        if match:
            try:
                metrics["nsys_total_gpu_time_ms"] = float(match.group(1))
            except (ValueError, IndexError):
                pass
        
        return metrics
    
    def _extract_ncu_metrics(self, ncu_rep_path: Path) -> Dict[str, float]:
        """Extract metrics from ncu report file."""
        metrics = {}
        
        if not ncu_rep_path.exists():
            return metrics
        
        # Try using ncu CLI to export metrics
        try:
            result = subprocess.run(
                ["ncu", "--csv", "--page", "details", "--import", str(ncu_rep_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                csv_metrics = self._parse_ncu_csv(result.stdout)
                metrics.update(csv_metrics)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        # Also check for companion CSV file
        companion_csv = ncu_rep_path.with_suffix(".csv")
        if companion_csv.exists():
            try:
                csv_text = companion_csv.read_text()
                csv_metrics = self._parse_ncu_csv(csv_text)
                metrics.update(csv_metrics)
            except Exception:
                pass
        
        return metrics
    
    def _parse_ncu_csv(self, csv_text: str) -> Dict[str, float]:
        """Parse ncu CSV output for key metrics."""
        metrics = {}
        
        # Look for key metrics in CSV
        patterns = {
            "ncu_sm_throughput_percent": r"sm__throughput.*?,(\d+\.?\d*)",
            "ncu_dram_throughput_percent": r"dram__throughput.*?,(\d+\.?\d*)",
            "ncu_tensor_core_utilization": r"tensor.*?active.*?,(\d+\.?\d*)",
            "ncu_occupancy": r"achieved.*?occupancy.*?,(\d+\.?\d*)",
            "ncu_gpu_time_duration": r"gpu__time_duration.*?,(\d+\.?\d*)",
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, csv_text, re.IGNORECASE)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except (ValueError, IndexError):
                    pass
        
        return metrics
    
    def _benchmark_without_profiling(
        self, fn: Callable, config: BenchmarkConfig
    ) -> List[float]:
        """Benchmark without profiling."""
        if self.mode == BenchmarkMode.TRITON:
            return self._benchmark_triton(fn, config)
        elif self.mode == BenchmarkMode.PYTORCH:
            return self._benchmark_pytorch(fn, config)
        else:
            return self._benchmark_custom(fn, config)
    
    def _benchmark_triton(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use Triton's do_bench (returns single value per call)."""
        try:
            import triton.testing as tt
            times_ms = []
            # Triton do_bench handles warmup internally, but we do our own
            for _ in range(config.iterations):
                time_ms = tt.do_bench(fn, warmup=0, rep=1)  # We handle warmup
                times_ms.append(time_ms)
            return times_ms
        except ImportError:
            # Fallback to custom if Triton not available
            return self._benchmark_custom(fn, config)
    
    def _benchmark_pytorch(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Use PyTorch's Timer."""
        try:
            from torch.utils.benchmark import Timer
            
            timer = Timer(
                stmt=fn,
                globals={},
                num_threads=1,
                device=self.device.type,
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
            return self._benchmark_custom(fn, config)
    
    def _benchmark_custom(self, fn: Callable, config: BenchmarkConfig) -> List[float]:
        """Custom benchmarking with CUDA Events for accurate GPU timing.
        
        Optimized for minimal overhead:
        - Uses CUDA Events for accurate GPU timing without blocking
        - Reuses events across iterations for efficiency
        - Synchronizes only when necessary for accurate timing
        """
        times_ms = []
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
                # Execute function under test
                fn()
                # Record end event (non-blocking)
                end_event.record()
                # Synchronize to ensure events are recorded, then get elapsed time
                # Note: Sync is necessary for accurate timing, but CUDA Events minimize overhead
                torch.cuda.synchronize(self.device)
                times_ms.append(start_event.elapsed_time(end_event))
        else:
            # CPU: use high-resolution timer
            for _ in range(config.iterations):
                start_time = time.perf_counter()
                fn()
                end_time = time.perf_counter()
                times_ms.append((end_time - start_time) * 1000)
        
        return times_ms
    
    def _warmup(self, fn: Callable, warmup_iterations: int) -> None:
        """Perform warmup iterations."""
        is_cuda = self.device.type == "cuda"
        for _ in range(warmup_iterations):
            fn()
        if is_cuda:
            torch.cuda.synchronize(self.device)
    
    def _compute_stats(
        self, times_ms: List[float], config: BenchmarkConfig
    ) -> BenchmarkResult:
        """Compute statistical measures."""
        if not times_ms:
            raise ValueError("No timing data collected")
        
        sorted_times = sorted(times_ms)
        n = len(sorted_times)
        
        # Compute percentiles
        percentiles_dict = {}
        for p in config.percentiles:
            idx = int((p / 100.0) * (n - 1))
            idx = min(idx, n - 1)
            percentiles_dict[p] = sorted_times[idx]
        
        return BenchmarkResult(
            mean_ms=statistics.mean(times_ms),
            median_ms=statistics.median(times_ms),
            std_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            percentiles=percentiles_dict,
            iterations=n,
            warmup_iterations=config.warmup,
        )


def compare_benchmarks(
    baseline: Benchmark,
    optimized: Benchmark,
    harness: Optional[BenchmarkHarness] = None,
    name: str = "Comparison"
) -> Dict[str, any]:
    """Compare baseline vs optimized benchmarks and return metrics."""
    if harness is None:
        harness = BenchmarkHarness()
    
    baseline_result = harness.benchmark(baseline)
    optimized_result = harness.benchmark(optimized)
    
    speedup = baseline_result.mean_ms / optimized_result.mean_ms if optimized_result.mean_ms > 0 else 1.0
    
    return {
        "name": name,
        "baseline": {
            "mean_ms": baseline_result.mean_ms,
            "median_ms": baseline_result.median_ms,
            "std_ms": baseline_result.std_ms,
            "min_ms": baseline_result.min_ms,
            "max_ms": baseline_result.max_ms,
        },
        "optimized": {
            "mean_ms": optimized_result.mean_ms,
            "median_ms": optimized_result.median_ms,
            "std_ms": optimized_result.std_ms,
            "min_ms": optimized_result.min_ms,
            "max_ms": optimized_result.max_ms,
        },
        "speedup": speedup,
        "baseline_result": baseline_result,
        "optimized_result": optimized_result,
    }

