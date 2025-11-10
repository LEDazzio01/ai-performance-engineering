"""Standard template and utilities for chapter compare.py modules.

All chapters should use these functions to ensure consistency:
- discover_benchmarks() - Find baseline/optimized pairs
- load_benchmark() - Load Benchmark instances from files
- create_profile_template() - Standard profile() function structure

All compare.py modules must:
1. Import BenchmarkHarness, Benchmark, BenchmarkMode, BenchmarkConfig
2. Use discover_benchmarks() to find pairs
3. Use load_benchmark() to instantiate benchmarks
4. Run via harness.benchmark(benchmark_instance)
5. Return standardized format: {"metrics": {...}}
"""

from __future__ import annotations

import importlib.util
import threading
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable, cast

import warnings
import torch

# Suppress CUDA capability warnings (GPU is newer than officially supported but works fine)
warnings.filterwarnings("ignore", message=".*Found GPU.*cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability.*", category=UserWarning)

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkHarness,
    BenchmarkMode,
    BenchmarkConfig,
)
from common.python.discovery import discover_benchmarks

# Import logger
try:
    from common.python.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ['discover_benchmarks', 'load_benchmark', 'create_standard_metrics', 'profile_template']


def load_benchmark(module_path: Path, timeout_seconds: int = 15) -> Optional[Benchmark]:
    """Load benchmark from module by calling get_benchmark() function.
    
    Uses threading timeout to prevent hangs during module import or get_benchmark() calls.
    
    Args:
        module_path: Path to Python file with Benchmark implementation
        timeout_seconds: Maximum time to wait for module load (default: 15 seconds)
        
    Returns:
        Benchmark instance or None if loading fails or times out
    """
    result: Dict[str, Any] = {"benchmark": None, "error": None, "done": False}
    
    def load_internal():
        try:
            # Add repo root (directory containing common/python) to sys.path
            repo_root = module_path.resolve()
            while repo_root.parent != repo_root:
                if (repo_root / "common" / "python").exists():
                    break
                repo_root = repo_root.parent
            else:
                repo_root = module_path.parent.parent
            
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            
            try:
                rel_path = module_path.resolve().relative_to(repo_root).with_suffix("")
                module_name = ".".join(rel_path.parts)
            except ValueError:
                module_name = module_path.stem
            
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                result["error"] = "Failed to create module spec"
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            simple_name = module_path.stem
            # Ensure module is discoverable via both its fully-qualified name and simple stem.
            # inspect.getmodule() checks sys.modules, so register both keys if missing.
            if module_name and module_name not in sys.modules:
                sys.modules[module_name] = module
            if simple_name not in sys.modules:
                sys.modules[simple_name] = module
            
            if hasattr(module, 'get_benchmark'):
                result["benchmark"] = module.get_benchmark()
            else:
                result["error"] = "Module does not have get_benchmark() function"
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True
    
    # Run load in a thread with timeout to prevent hangs
    thread = threading.Thread(target=load_internal, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if not result["done"]:
        logger.warning(f"Failed to load {module_path.name}: TIMEOUT (exceeded {timeout_seconds}s)")
        return None
    
    if result["error"]:
        logger.warning(f"Failed to load {module_path.name}: {result['error']}")
        return None
    
    return cast(Optional[Benchmark], result["benchmark"])


def create_standard_metrics(
    chapter: str,
    all_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Create standardized metrics dictionary from collected results.
    
    Ensures all chapters return consistent metrics format.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch1', 'ch16')
        all_metrics: Dictionary of collected metrics (will be modified in place)
        default_tokens_per_s: Default throughput if not calculated
        default_requests_per_s: Default request rate if not calculated
        default_goodput: Default efficiency metric if not calculated
        default_latency_s: Default latency if not calculated
        
    Returns:
        Standardized metrics dictionary
    """
    # Ensure chapter is set
    all_metrics['chapter'] = chapter
    
    # Calculate speedups from collected metrics
    speedups = [
        v for k, v in all_metrics.items() 
        if k.endswith('_speedup') and isinstance(v, (int, float)) and v > 0
    ]
    
    if speedups:
        all_metrics['speedup'] = max(speedups)
        all_metrics['average_speedup'] = sum(speedups) / len(speedups)
    else:
        # Default if no speedups found
        all_metrics['speedup'] = 1.0
        all_metrics['average_speedup'] = 1.0
    
    return all_metrics


def profile_template(
    chapter: str,
    chapter_dir: Path,
    harness_config: Optional[BenchmarkConfig] = None,
    custom_metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Template profile() function for chapter compare.py modules.
    
    Standard implementation that all chapters should use or adapt.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch1', 'ch16')
        chapter_dir: Path to chapter directory
        harness_config: Optional BenchmarkConfig override (default: iterations=20, warmup=5)
        custom_metrics_callback: Optional function to add custom metrics: f(all_metrics) -> None
        
    Returns:
        Standardized format: {"metrics": {...}}
    """
    logger.info("=" * 70)
    logger.info(f"Chapter {chapter.upper()}: Comparing Implementations")
    logger.info("=" * 70)
    
    chapter_tokens: List[float] = []
    chapter_requests: List[float] = []
    chapter_latencies_ms: List[float] = []
    chapter_goodputs: List[float] = []

    def record_throughput(prefix: str, throughput_obj: Optional[Any], metrics: Dict[str, Any]) -> None:
        if not throughput_obj:
            return
        tokens = getattr(throughput_obj, "tokens_per_s", None)
        requests = getattr(throughput_obj, "requests_per_s", None)
        samples = getattr(throughput_obj, "samples_per_s", None)
        bytes_per_s = getattr(throughput_obj, "bytes_per_s", None)
        custom_unit = getattr(throughput_obj, "custom_unit_per_s", None)
        custom_unit_name = getattr(throughput_obj, "custom_unit_name", None)
        latency_ms = getattr(throughput_obj, "latency_ms", None)
        goodput = getattr(throughput_obj, "goodput", None)

        if tokens is not None:
            metrics[f"{prefix}_tokens_per_s"] = tokens
            chapter_tokens.append(tokens)
        if requests is not None:
            metrics[f"{prefix}_requests_per_s"] = requests
            chapter_requests.append(requests)
        if samples is not None:
            metrics[f"{prefix}_samples_per_s"] = samples
        if bytes_per_s is not None:
            metrics[f"{prefix}_bytes_per_s"] = bytes_per_s
        if custom_unit is not None:
            key = f"{prefix}_custom_unit_per_s"
            metrics[key] = custom_unit
            if custom_unit_name:
                metrics[f"{prefix}_custom_unit_name"] = custom_unit_name
        if latency_ms is not None:
            metrics[f"{prefix}_latency_ms"] = latency_ms
            metrics[f"{prefix}_latency_s"] = latency_ms / 1000.0
            chapter_latencies_ms.append(latency_ms)
        if goodput is not None:
            metrics[f"{prefix}_goodput"] = goodput
            chapter_goodputs.append(goodput)

    def format_throughput_summary(throughput_obj: Optional[Any]) -> str:
        if not throughput_obj:
            return ""
        parts = []
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
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping")
        return {
            "metrics": {
                'chapter': chapter,
                'cuda_unavailable': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    # Initialize CUDA context early to prevent cuBLAS warnings
    # This ensures the context is ready before any operations
    try:
        torch.cuda.init()
        # Create a dummy tensor to force context initialization
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"CUDA context initialization warning (non-fatal): {e}")
    
    pairs = discover_benchmarks(chapter_dir)
    
    if not pairs:
        logger.warning("No baseline/optimized pairs found")
        logger.info("Tip: Create baseline_*.py and optimized_*.py files")
        logger.info("    Each file must implement Benchmark protocol with get_benchmark() function")
        return {
            "metrics": {
                'chapter': chapter,
                'no_pairs_found': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    logger.info(f"Found {len(pairs)} example(s) with optimization(s):\n")
    
    # Create harness with default or custom config
    # Enable memory tracking by default to capture all metrics
    # Increase timeouts to prevent premature failures
    from dataclasses import replace
    if harness_config is None:
        config = BenchmarkConfig(
            iterations=20, 
            warmup=5, 
            enable_memory_tracking=True,
            measurement_timeout_seconds=30,  # Increased from 15 to prevent timeouts
            setup_timeout_seconds=60,  # Increased for slow setups
        )
    else:
        # Enable memory tracking by default unless explicitly disabled
        # Increase timeouts if not explicitly set
        if harness_config.enable_memory_tracking is False:
            # Respect explicit False
            config = harness_config
        else:
            # Default to True for comprehensive metrics
            config = replace(harness_config, enable_memory_tracking=True)
        
        # Ensure reasonable timeouts
        if config.measurement_timeout_seconds < 30:
            config.measurement_timeout_seconds = 30
        if config.setup_timeout_seconds is not None and config.setup_timeout_seconds < 60:
            config.setup_timeout_seconds = 60
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    
    all_metrics: Dict[str, Any] = {
        'chapter': chapter,
    }
    
    # Collect all results for summary
    summary_data: List[Dict[str, Any]] = []
    
    for baseline_path, optimized_paths, example_name in pairs:
        baseline_benchmark = load_benchmark(baseline_path)
        if baseline_benchmark is None:
            logger.error(f"Baseline failed to load (missing get_benchmark() function?): {baseline_path.name}")
            continue
        
        try:
            baseline_result = harness.benchmark(baseline_benchmark)
            baseline_timing = baseline_result.timing
            baseline_time = baseline_timing.mean_ms if baseline_timing else 0.0
            
            # Display comprehensive baseline metrics
            logger.info(f"  Example: {example_name}")
            logger.info(f"    Baseline: {baseline_path.name}")
            logger.info(f"    Baseline: {baseline_time:.2f} ms")
            if baseline_timing:
                logger.info(f"      📊 Timing Stats: median={baseline_timing.median_ms:.2f}ms, "
                      f"min={baseline_timing.min_ms:.2f}ms, max={baseline_timing.max_ms:.2f}ms, "
                      f"std={baseline_timing.std_ms:.2f}ms")
            if baseline_result.memory:
                mem_str = f"      💾 Memory: peak={baseline_result.memory.peak_mb:.2f}MB"
                if baseline_result.memory.allocated_mb is not None:
                    mem_str += f", allocated={baseline_result.memory.allocated_mb:.2f}MB"
                logger.info(mem_str)
            if baseline_timing and baseline_timing.percentiles:
                p99 = baseline_timing.percentiles.get(99.0)
                if p99:
                    logger.info(f"      📈 Percentiles: p99={p99:.2f}ms, p75={baseline_timing.percentiles.get(75.0, 0):.2f}ms, "
                          f"p50={baseline_timing.percentiles.get(50.0, 0):.2f}ms")
            baseline_throughput = getattr(baseline_result, "throughput", None)
            throughput_summary = format_throughput_summary(baseline_throughput)
            if throughput_summary:
                logger.info(f"      ⚡ Throughput: {throughput_summary}")
            record_throughput(f"{example_name}_baseline", baseline_throughput, all_metrics)
        except Exception as e:
            error_msg = str(e)
            # Check for skip warnings
            if "SKIPPED" in error_msg or "SKIP" in error_msg.upper() or "WARNING: SKIPPED" in error_msg:
                logger.warning(f"{error_msg}")
            else:
                logger.error(f"Baseline failed to run: {error_msg}")
            continue
        
        best_speedup = 1.0
        best_optimized = None
        optimized_results = []
        
        for optimized_path in optimized_paths:
            opt_name = optimized_path.name
            # Extract technique name: optimized_moe_sparse.py -> sparse
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.py', '')
            if technique == opt_name.replace('optimized_', '').replace('.py', ''):
                technique = 'default'
            
            optimized_benchmark = load_benchmark(optimized_path)
            if optimized_benchmark is None:
                logger.error(f"Failed to load: {opt_name}")
                continue
            
            try:
                optimized_result = harness.benchmark(optimized_benchmark)
                optimized_timing = optimized_result.timing
                optimized_time = optimized_timing.mean_ms if optimized_timing else 0.0
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                
                # Format output matching user's example: "0.06 ms (4.97x)"
                if speedup >= 1.0:
                    speedup_str = f"{optimized_time:.2f} ms ({speedup:.2f}x) 🚀"
                else:
                    speedup_str = f"{optimized_time:.2f} ms ({speedup:.2f}x) ⚠️"
                
                logger.info(f"    Testing: {opt_name}... {speedup_str}")
                
                # Display comprehensive optimized metrics
                if optimized_timing:
                    logger.info(f"        📊 Timing: median={optimized_timing.median_ms:.2f}ms, "
                          f"min={optimized_timing.min_ms:.2f}ms, max={optimized_timing.max_ms:.2f}ms, "
                          f"std={optimized_timing.std_ms:.2f}ms")
                
                # Memory comparison
                if optimized_result.memory:
                    mem_change = ""
                    opt_peak = optimized_result.memory.peak_mb
                    base_peak = baseline_result.memory.peak_mb if baseline_result.memory else None
                    if opt_peak is not None and base_peak is not None:
                        mem_diff = opt_peak - base_peak
                        mem_change_pct = (mem_diff / base_peak * 100) if base_peak > 0 else 0
                        if mem_diff > 0:
                            mem_change = f" (+{mem_diff:.2f}MB, +{mem_change_pct:.1f}%)"
                        elif mem_diff < 0:
                            mem_change = f" ({mem_diff:.2f}MB, {mem_change_pct:.1f}%)"
                        else:
                            mem_change = " (no change)"
                    
                    if opt_peak is not None:
                        logger.info(f"        💾 Memory: peak={opt_peak:.2f}MB{mem_change}")
                    if optimized_result.memory.allocated_mb is not None:
                        logger.info(f"                 allocated={optimized_result.memory.allocated_mb:.2f}MB")
                
                optimized_throughput = getattr(optimized_result, "throughput", None)
                throughput_summary = format_throughput_summary(optimized_throughput)
                if throughput_summary:
                    logger.info(f"        ⚡ Throughput: {throughput_summary}")
                record_throughput(f"{example_name}_{technique}", optimized_throughput, all_metrics)
                
                # Percentile comparison
                if optimized_timing and optimized_timing.percentiles and baseline_timing and baseline_timing.percentiles:
                    p99_opt = optimized_timing.percentiles.get(99.0)
                    p99_base = baseline_timing.percentiles.get(99.0)
                    if p99_opt and p99_base:
                        p99_speedup = p99_base / p99_opt if p99_opt > 0 else 1.0
                        logger.info(f"        📈 Percentiles: p99={p99_opt:.2f}ms ({p99_speedup:.2f}x), "
                              f"p75={optimized_timing.percentiles.get(75.0, 0):.2f}ms, "
                              f"p50={optimized_timing.percentiles.get(50.0, 0):.2f}ms")
                
                # Visual bar chart for speedup (inline)
                bar_width = 40
                if speedup >= 1.0:
                    filled = int(min(bar_width, (speedup - 1.0) / 4.0 * bar_width))
                    bar = "█" * filled + "░" * (bar_width - filled)
                else:
                    filled = int(min(bar_width, (1.0 - speedup) / 0.5 * bar_width))
                    bar = "░" * (bar_width - filled) + "█" * filled
                
                logger.info(f"        [{bar}] {speedup:.2f}x speedup")
                
                optimized_results.append({
                    'name': opt_name,
                    'time': optimized_time,
                    'speedup': speedup,
                    'technique': technique
                })
                
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_optimized = optimized_path
                
                # Store per-technique metrics
                all_metrics[f"{example_name}_{technique}_baseline_time"] = baseline_time
                all_metrics[f"{example_name}_{technique}_optimized_time"] = optimized_time
                all_metrics[f"{example_name}_{technique}_speedup"] = speedup
                
            except Exception as e:
                error_msg = str(e)
                # Check for skip warnings
                if "SKIPPED" in error_msg or "SKIP" in error_msg.upper():
                    logger.warning(f"{error_msg}")
                else:
                    logger.error(f"Failed: {error_msg}")
                continue
        
        # Summary for this example
        if optimized_results:
            summary_data.append({
                'example': example_name,
                'baseline': baseline_time,
                'best_speedup': best_speedup,
                'best_name': best_optimized.name if best_optimized else None,
                'num_optimizations': len(optimized_results)
            })
            all_metrics[f"{example_name}_best_speedup"] = best_speedup
        
        if not optimized_results:
            summary_data.append({
                'example': example_name,
                'baseline': baseline_time,
                'best_speedup': 1.0,
                'best_name': None,
                'num_optimizations': 0
            })
    
    if chapter_requests:
        all_metrics['requests_per_s'] = max(chapter_requests)
    if chapter_tokens:
        all_metrics['tokens_per_s'] = max(chapter_tokens)
    if chapter_latencies_ms:
        best_latency = min(chapter_latencies_ms)
        all_metrics['latency_ms'] = best_latency
        all_metrics['latency_s'] = best_latency / 1000.0
    if chapter_goodputs:
        all_metrics['goodput'] = max(chapter_goodputs)

    # Apply custom metrics callback if provided
    if custom_metrics_callback is not None:
        custom_metrics_callback(all_metrics)
    
    # Standardize metrics format
    all_metrics = create_standard_metrics(chapter, all_metrics)
    
    # Print snazzy summary
    if summary_data:
        print("\n" + "=" * 80)
        print("📊 SUMMARY - Performance Improvements")
        print("=" * 80)
        
        # Sort by speedup (best first)
        def get_speedup(x: Dict[str, Any]) -> float:
            val = x.get('best_speedup', 1.0)
            return float(val) if isinstance(val, (int, float)) else 1.0
        
        summary_data.sort(key=get_speedup, reverse=True)
        
        for idx, item in enumerate(summary_data, 1):
            example_name = str(item.get('example', ''))
            baseline_val = item.get('baseline', 0.0)
            baseline = float(baseline_val) if isinstance(baseline_val, (int, float)) else 0.0
            speedup_val = item.get('best_speedup', 1.0)
            speedup = float(speedup_val) if isinstance(speedup_val, (int, float)) else 1.0
            best_name_val = item.get('best_name')
            best_name = str(best_name_val) if best_name_val is not None else None
            num_opts_val = item.get('num_optimizations', 0)
            num_opts = int(num_opts_val) if isinstance(num_opts_val, (int, float)) else 0
            
            # Status emoji
            if speedup >= 2.0:
                status = "🔥"
            elif speedup >= 1.5:
                status = "✨"
            elif speedup >= 1.2:
                status = "👍"
            elif speedup >= 1.0:
                status = "✅"
            else:
                status = "⚠️"
            
            logger.info(f"\n  {idx}. {example_name} {status}")
            logger.info(f"     Baseline: {baseline:.2f} ms")
            
            if best_name and speedup > 1.0:
                improvement_pct = (1 - 1/speedup) * 100
                logger.info(f"     🏆 Best: {best_name}")
                logger.info(f"     📈 Improvement: {speedup:.2f}x faster ({improvement_pct:.1f}% reduction)")
                
                # ASCII bar showing improvement
                bar_width = 50
                filled = int(min(bar_width, (speedup - 1.0) / 5.0 * bar_width))
                bar = "█" * filled + "░" * (bar_width - filled)
                logger.info(f"     {bar}")
            elif num_opts == 0:
                logger.warning(f"     ⚠️  No successful optimizations")
            else:
                logger.warning(f"     ⚠️  Optimization did not improve performance")
            
            logger.info(f"     📦 {num_opts} optimization(s) tested")
        
        # Overall stats
        def get_speedup_for_filter(s: Dict[str, Any]) -> bool:
            val = s.get('best_speedup', 1.0)
            return float(val) > 1.0 if isinstance(val, (int, float)) else False
        
        def get_speedup_for_sum(s: Dict[str, Any]) -> float:
            val = s.get('best_speedup', 1.0)
            return float(val) if isinstance(val, (int, float)) else 1.0
        
        successful = [s for s in summary_data if get_speedup_for_filter(s)]
        avg_speedup = sum(get_speedup_for_sum(s) for s in successful) / len(successful) if successful else 0
        best_overall = max(summary_data, key=get_speedup_for_sum)
        
        logger.info("\n" + "-" * 80)
        logger.info(f"📊 Overall Stats:")
        logger.info(f"   • Examples tested: {len(summary_data)}")
        logger.info(f"   • Successful optimizations: {len(successful)}")
        if successful:
            logger.info(f"   • Average speedup: {avg_speedup:.2f}x")
            logger.info(f"   • Best improvement: {best_overall['example']} ({best_overall['best_speedup']:.2f}x)")
        logger.info("=" * 80)
    
    logger.info("")
    
    return {
        "metrics": all_metrics
    }


__all__ = [
    'discover_benchmarks',
    'load_benchmark',
    'create_standard_metrics',
    'profile_template',
]
