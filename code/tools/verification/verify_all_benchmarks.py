#!/usr/bin/env python3
"""Verify all baseline/optimized benchmarks can be loaded and executed.

Tests:
1. All files compile (syntax check)
2. All benchmarks can be imported
3. All benchmarks can be instantiated via get_benchmark()
4. All benchmarks can run setup() without errors
5. All benchmarks can run benchmark_fn() without errors (minimal run)

NOTE: Distributed benchmarks are ONLY skipped if num_gpus == 1 (single GPU system).
This is clearly logged when it happens.

Usage:
    Prefer running via: python tools/cli/aisp bench.py verify [--targets ...]
"""

import sys
import os
import importlib.util
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Apply environment defaults (creates .torch_inductor directory, etc.)
try:
    from common.python.env_defaults import apply_env_defaults
    apply_env_defaults()
except ImportError:
    pass  # Continue if env_defaults not available

# Default timeout constant (15 seconds - required for all benchmarks)
DEFAULT_TIMEOUT = 15

EXAMPLE_ALIASES = {}

from common.python.discovery import (
    chapter_slug,
    discover_all_chapters,
    normalize_chapter_token,
)


def apply_example_alias(chapter: str, example: str) -> str:
    """Aliases disabled; return example unchanged."""
    return example


def normalize_chapter_name(raw: str) -> str:
    return normalize_chapter_token(raw)


def normalize_example_name(raw: str) -> str:
    example = raw.strip()
    if not example:
        raise ValueError("Example name cannot be empty.")
    if example.endswith('.py'):
        example = example[:-3]
    for prefix in ('baseline_', 'optimized_'):
        if example.startswith(prefix):
            example = example[len(prefix):]
            break
    if not example:
        raise ValueError(f"Invalid example identifier '{raw}'.")
    return example


def resolve_target_chapters(target_args: Optional[List[str]]) -> Tuple[List[Path], Dict[str, Optional[Set[str]]]]:
    """Return ordered chapter directories and per-chapter example filters.
    
    Args:
        target_args: List of chapter or chapter:example tokens. Use None or ['all']
            to select every chapter.
    """
    if not target_args:
        chapter_dirs = discover_all_chapters(repo_root)
        if not chapter_dirs:
            raise FileNotFoundError("No chapter directories found.")
        chapter_filters = {chapter_slug(d, repo_root): None for d in chapter_dirs}
        return chapter_dirs, chapter_filters
    
    normalized = [token.strip().lower() for token in target_args]
    if 'all' in normalized:
        if len(normalized) > 1:
            raise ValueError("'all' cannot be combined with other targets.")
        return resolve_target_chapters(None)
    
    chapter_order, chapter_filters = parse_target_args(target_args)
    chapter_dirs: List[Path] = []
    for chapter_name in chapter_order:
        chapter_dir = repo_root / chapter_name
        if not chapter_dir.exists():
            raise FileNotFoundError(f"Chapter directory '{chapter_name}' not found.")
        chapter_dirs.append(chapter_dir)
    return chapter_dirs, chapter_filters


def parse_target_args(target_args: List[str]) -> Tuple[List[str], Dict[str, Optional[Set[str]]]]:
    """Return chapter order and mapping from chapter -> set of example names (or None for all)."""
    chapter_order: List[str] = []
    chapter_filters: Dict[str, Optional[Set[str]]] = {}
    for raw_target in target_args:
        chapter_part, example_part = (raw_target.split(':', 1) + [None])[:2]
        chapter = normalize_chapter_name(chapter_part)
        if chapter not in chapter_filters:
            chapter_order.append(chapter)
            chapter_filters[chapter] = set()
        if example_part is None or example_part.strip() == '':
            chapter_filters[chapter] = None  # Whole chapter requested; overrides specifics
            continue
        if chapter_filters[chapter] is None:
            continue  # Already targeting entire chapter; ignore narrower filters
        normalized_example = normalize_example_name(example_part)
        normalized_example = apply_example_alias(chapter, normalized_example)
        chapter_filters[chapter].add(normalized_example)
    return chapter_order, chapter_filters


def check_syntax(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), str(file_path), 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Compile error: {e}"


def load_benchmark(file_path: Path, timeout_seconds: int = DEFAULT_TIMEOUT) -> Tuple[Optional[object], Optional[str]]:
    """Load benchmark from file and return instance.
    
    Uses threading timeout to prevent hangs during module import or get_benchmark() calls.
    
    Note: For benchmarks that compile CUDA extensions during import (rare), the default
    15-second timeout may be insufficient. Most CUDA extensions are lazy-loaded in setup(),
    but if compilation happens during import, consider pre-compiling extensions or increasing
    the timeout parameter.
    
    Args:
        file_path: Path to Python file with Benchmark implementation
        timeout_seconds: Maximum time to wait for module load (default: 15 seconds)
                        Increase to 60-120 seconds if CUDA compilation happens during import
        
    Returns:
        Tuple of (benchmark_instance, error_message). If successful: (benchmark, None).
        If failed or timed out: (None, error_string).
    """
    import threading
    
    result = {"benchmark": None, "error": None, "done": False}
    
    def load_internal():
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                result["error"] = "Could not create module spec"
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'get_benchmark'):
                result["error"] = "Missing get_benchmark() function"
                return
            
            result["benchmark"] = module.get_benchmark()
        except Exception as e:
            result["error"] = f"Load error: {e}"
        finally:
            result["done"] = True
    
    # Run load in a thread with timeout to prevent hangs
    thread = threading.Thread(target=load_internal, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if not result["done"]:
        return None, f"TIMEOUT: exceeded {timeout_seconds} second timeout (module import/get_benchmark() took too long)"
    
    if result["error"]:
        return None, result["error"]
    
    return result["benchmark"], None


def test_benchmark(benchmark: object, timeout: int = DEFAULT_TIMEOUT) -> Tuple[bool, Optional[str]]:
    """Test benchmark execution with timeout protection.
    
    Runs full execution: setup(), benchmark_fn(), teardown()
    Resets CUDA state before and after to prevent cascading failures.
    
    Uses threading timeout (reliable, cross-platform) instead of signal-based timeout.
    
    If benchmark has get_config() method, uses setup_timeout_seconds from config if available,
    otherwise falls back to provided timeout.
    """
    import threading
    import torch
    
    # Check if benchmark specifies longer timeouts in its config
    # Note: We use the maximum of all timeouts since test_benchmark() runs setup + benchmark_fn + teardown
    # as a single operation, so we need to account for the longest phase
    original_timeout = timeout
    if hasattr(benchmark, 'get_config'):
        try:
            config = benchmark.get_config()
            # Check setup timeout (most relevant for CUDA compilation)
            if hasattr(config, 'setup_timeout_seconds') and config.setup_timeout_seconds:
                timeout = max(timeout, config.setup_timeout_seconds)
            # Also check measurement timeout (for long-running benchmark_fn)
            if hasattr(config, 'measurement_timeout_seconds') and config.measurement_timeout_seconds:
                timeout = max(timeout, config.measurement_timeout_seconds)
            # Check warmup timeout (less relevant for single-run verification, but included for completeness)
            if hasattr(config, 'warmup_timeout_seconds') and config.warmup_timeout_seconds:
                timeout = max(timeout, config.warmup_timeout_seconds)
        except Exception:
            # If get_config() fails, use default timeout
            pass
    
    # Note: If timeout was increased from config, extensions should be pre-compiled
    # Threading timeout may not interrupt CUDA compilation, so pre-compilation is recommended
    
    def reset_cuda_state():
        """Reset CUDA state to prevent cascading failures."""
        try:
            if torch.cuda.is_available():
                # Synchronize to catch any pending CUDA errors
                torch.cuda.synchronize()
                # Clear any device-side errors
                torch.cuda.reset_peak_memory_stats()
                # Clear cache
                torch.cuda.empty_cache()
                # Synchronize again after cleanup
                torch.cuda.synchronize()
        except RuntimeError:
            # If synchronization fails, try to reset device
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Reset CUDA state before running benchmark
    reset_cuda_state()
    
    execution_result = {"success": False, "error": None, "done": False}
    
    def run_benchmark():
        """Run benchmark in a separate thread with timeout protection."""
        try:
            # Test setup
            if hasattr(benchmark, 'setup'):
                benchmark.setup()
            
            # Test benchmark_fn (full execution)
            if hasattr(benchmark, 'benchmark_fn'):
                benchmark.benchmark_fn()
            
            # Test teardown (no timeout needed, should be fast)
            if hasattr(benchmark, 'teardown'):
                benchmark.teardown()
            
            # Reset CUDA state after successful execution
            reset_cuda_state()
            
            # Only mark as success if we got here without exceptions
            execution_result["success"] = True
        except Exception as e:
            reset_cuda_state()  # Reset on error to prevent cascading failures
            execution_result["error"] = e
            execution_result["success"] = False  # Explicitly mark as failed
        finally:
            execution_result["done"] = True
    
    # Run benchmark in thread with timeout (required, default 15 seconds)
    # Only print timeout message if timeout actually occurs (not upfront)
    thread = threading.Thread(target=run_benchmark, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if not execution_result["done"]:
        # TIMEOUT OCCURRED - make it very clear
        print("\n" + "=" * 80)
        print("TIMEOUT: Benchmark execution exceeded timeout limit")
        print("=" * 80)
        print(f"   Timeout limit: {timeout} seconds")
        print(f"   Status: Benchmark did not complete within timeout period")
        print(f"   Action: Benchmark execution was terminated to prevent hang")
        print("=" * 80)
        print()
        
        reset_cuda_state()  # Reset on timeout too
        return False, f"TIMEOUT: exceeded {timeout} second timeout"
    
    if execution_result["error"]:
        # Error occurred during execution
        error = execution_result["error"]
        return False, f"Execution error: {str(error)}\n{traceback.format_exc()}"
    
    # Don't print success message for normal completion - only print on timeout/failure
    if execution_result["success"]:
        return True, None
    
    # Shouldn't reach here, but handle gracefully
    return False, "Unknown error during benchmark execution"


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


def verify_chapter(chapter_dir: Path, target_examples: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Verify all benchmarks in a chapter.
    
    Runs ALL tests. Only skips distributed benchmarks if num_gpus == 1,
    and logs this clearly.
    """
    import torch

    chapter_id = chapter_slug(chapter_dir, repo_root)
    results = {
        'chapter': chapter_id,
        'total': 0,
        'syntax_pass': 0,
        'load_pass': 0,
        'exec_pass': 0,
        'skipped': [],
        'failures': []
    }
    
    # Check GPU count for distributed benchmark detection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Find all baseline and optimized files
    baseline_files = list(chapter_dir.glob("baseline_*.py"))
    optimized_files = list(chapter_dir.glob("optimized_*.py"))
    all_files: List[Path]
    if target_examples is None:
        all_files = baseline_files + optimized_files
    else:
        baseline_index = {
            file_path.stem[len("baseline_"):]: file_path
            for file_path in baseline_files
            if file_path.stem.startswith("baseline_")
        }
        optimized_index = {
            file_path.stem[len("optimized_"):]: file_path
            for file_path in optimized_files
            if file_path.stem.startswith("optimized_")
        }
        missing_files: List[str] = []
        selected: List[Path] = []
        for example in sorted(target_examples):
            baseline_key = f"baseline_{example}.py"
            optimized_key = f"optimized_{example}.py"
            baseline_path = baseline_index.get(example)
            optimized_path = optimized_index.get(example)
            if baseline_path is None:
                missing_files.append(baseline_key)
            else:
                selected.append(baseline_path)
            if optimized_path is None:
                missing_files.append(optimized_key)
            else:
                selected.append(optimized_path)
        if missing_files:
            missing_csv = ", ".join(missing_files)
            raise FileNotFoundError(f"{chapter_id}: missing {missing_csv}")
        all_files = selected
    
    if not all_files:
        raise FileNotFoundError(f"{chapter_id}: no benchmark files found matching the requested targets")
    
    results['total'] = len(all_files)
    
    for file_path in sorted(all_files):
        file_name = file_path.name
        
        # Check syntax
        syntax_ok, syntax_err = check_syntax(file_path)
        if not syntax_ok:
            results['failures'].append({
                'file': file_name,
                'stage': 'syntax',
                'error': syntax_err
            })
            continue
        results['syntax_pass'] += 1
        
        # Load benchmark
        benchmark, load_err = load_benchmark(file_path)
        if benchmark is None:
            if load_err and "SKIPPED:" in load_err:
                results['skipped'].append({
                    'file': file_name,
                    'reason': load_err
                })
                print(f"    WARNING: {file_name}: {load_err}")
            else:
                results['failures'].append({
                    'file': file_name,
                    'stage': 'load',
                    'error': load_err
                })
            continue
        results['load_pass'] += 1
        
        # Check if this is a distributed benchmark and we have only 1 GPU
        is_distributed = is_distributed_benchmark(file_path)
        if is_distributed and num_gpus == 1:
            # SKIP ONLY when distributed benchmark on single GPU system
            skip_reason = f"SKIPPED: Distributed benchmark requires multiple GPUs (found {num_gpus} GPU)"
            results['skipped'].append({
                'file': file_name,
                'reason': skip_reason
            })
            print(f"    WARNING: {file_name}: {skip_reason}")
            results['exec_pass'] += 1  # Count as pass since we intentionally skipped
            continue
        
        # Test execution (ALL benchmarks run - no skipping except single-GPU distributed)
        exec_ok, exec_err = test_benchmark(benchmark, timeout=DEFAULT_TIMEOUT)
        if not exec_ok:
            if exec_err and "SKIPPED:" in exec_err:
                results['skipped'].append({
                    'file': file_name,
                    'reason': exec_err
                })
                print(f"    WARNING: {file_name}: {exec_err}")
            else:
                results['failures'].append({
                    'file': file_name,
                    'stage': 'execution',
                    'error': exec_err
                })
            continue
        results['exec_pass'] += 1
    
    return results


def run_verification(target_args: Optional[List[str]] = None) -> int:
    import torch
    import subprocess
    
    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(target_args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    
    print("=" * 80)
    print("VERIFYING ALL BASELINE/OPTIMIZED BENCHMARKS")
    print("=" * 80)
    print("Mode: FULL EXECUTION - All tests run")
    print()
    
    # Check system configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"System: {num_gpus} GPU(s) available")
        if num_gpus == 1:
            print("WARNING: NOTE: Distributed benchmarks will be SKIPPED (require multiple GPUs)")
            print("   This will be clearly logged for each skipped benchmark")
        
        # Pre-compile CUDA extensions to avoid timeout issues during verification
        print("\nPre-compiling CUDA extensions to avoid timeout issues...")
        try:
            precompile_path = repo_root / "tools" / "utilities" / "precompile_cuda_extensions.py"
            if precompile_path.exists():
                result = subprocess.run(
                    [sys.executable, str(precompile_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes for compilation
                )
                if result.returncode == 0:
                    print("  [OK] CUDA extensions pre-compiled successfully")
                else:
                    print("  WARNING: Some CUDA extensions failed to pre-compile")
                    print("    They will compile at runtime (may cause timeouts)")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}")
            else:
                print("  INFO: Pre-compilation script not found - extensions will compile at runtime")
        except subprocess.TimeoutExpired:
            print("  WARNING: Pre-compilation timed out - extensions will compile at runtime")
        except Exception as e:
            print(f"  WARNING: Could not pre-compile extensions: {e}")
            print("    Extensions will compile at runtime (may cause timeouts)")
    else:
        print("System: No CUDA GPUs available")
        print("WARNING: NOTE: All GPU benchmarks will likely fail")
    print()
    
    all_results = []
    total_files = 0
    total_syntax_pass = 0
    total_load_pass = 0
    total_exec_pass = 0
    total_failures = 0
    
    for chapter_dir in chapter_dirs:
        if not chapter_dir.exists():
            continue

        chapter_id = chapter_slug(chapter_dir, repo_root)
        print(f"Testing {chapter_id}...")
        try:
            results = verify_chapter(chapter_dir, chapter_filters[chapter_id])
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1
        all_results.append(results)
        
        total_files += results['total']
        total_syntax_pass += results['syntax_pass']
        total_load_pass += results['load_pass']
        total_exec_pass += results['exec_pass']
        total_failures += len(results['failures'])
        total_skipped = sum(len(r['skipped']) for r in all_results)
        
        # Print chapter summary
        status = "PASS" if len(results['failures']) == 0 else "WARN"
        skipped_msg = f", {len(results['skipped'])} skipped" if results['skipped'] else ""
        print(f"  {status} {results['total']} files: "
              f"{results['syntax_pass']} syntax, "
              f"{results['load_pass']} load, "
              f"{results['exec_pass']} exec, "
              f"{len(results['failures'])} failures{skipped_msg}")
    
    # Calculate total skipped
    total_skipped = sum(len(r['skipped']) for r in all_results)
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files tested: {total_files}")
    print(f"Syntax check passed: {total_syntax_pass}/{total_files} ({100*total_syntax_pass/max(total_files,1):.1f}%)")
    print(f"Load check passed: {total_load_pass}/{total_files} ({100*total_load_pass/max(total_files,1):.1f}%)")
    print(f"Execution check passed: {total_exec_pass}/{total_files} ({100*total_exec_pass/max(total_files,1):.1f}%)")
    print(f"Total failures: {total_failures}")
    if total_skipped > 0:
        print(f"Total skipped: {total_skipped} (distributed benchmarks on single-GPU system)")
    print()
    
    # Print skipped benchmarks (EXTREMELY CLEAR)
    if total_skipped > 0:
        print("=" * 80)
        print("SKIPPED BENCHMARKS (Single-GPU System)")
        print("=" * 80)
        print("These benchmarks were SKIPPED because they require multiple GPUs")
        print(f"and this system has only {torch.cuda.device_count() if torch.cuda.is_available() else 0} GPU(s).")
        print()
        for results in all_results:
            if results['skipped']:
                print(f"{results['chapter']}:")
                for skipped in results['skipped']:
                    print(f"  WARNING: SKIPPED: {skipped['file']}")
                    print(f"     Reason: {skipped['reason']}")
        print()
    
    # Print failures
    if total_failures > 0:
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)
        for results in all_results:
            if results['failures']:
                print(f"\n{results['chapter']}:")
                for failure in results['failures']:
                    print(f"  FAILED: {failure['file']} ({failure['stage']}): {failure['error']}")
        print()
        return 1
    else:
        print("All benchmarks verified successfully!")
        if total_skipped > 0:
            print(f"(Note: {total_skipped} distributed benchmarks skipped on single-GPU system)")
        return 0


def main():
    raise SystemExit(
        "This entrypoint moved. Run 'python tools/cli/aisp bench.py verify [--targets ...]' instead."
    )


if __name__ == "__main__":
    sys.exit(main())
