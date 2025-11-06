#!/usr/bin/env python3
"""Run all benchmarks - discover, run, and summarize results.

Usage:
    python benchmark.py                           # Run all benchmarks
    python benchmark.py --chapter 12              # Run benchmarks for chapter 12 (accepts number or ch12)
    python benchmark.py --chapter ch12            # Run benchmarks for chapter 12 (accepts number or ch12)
    python benchmark.py --format json              # Output JSON only
    python benchmark.py --chapter 12 --format json # Run ch12, output JSON only
    python benchmark.py --profile                  # Enable nsys profiling (generates .nsys-rep files)
    python benchmark.py --chapter 1 --profile       # Run ch1 with profiling enabled
    python benchmark.py --suite-timeout 7200        # Set suite timeout to 2 hours (default: 4 hours)
    python benchmark.py --suite-timeout 0          # Disable suite timeout (run until completion)
"""

import sys
import warnings

# Suppress CUDA capability warnings for GB10 (12.1) - PyTorch supports up to 12.0
warnings.filterwarnings("ignore", message=".*Found GPU.*which is of cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)

from common.python.env_defaults import apply_env_defaults, dump_environment_and_capabilities

apply_env_defaults()
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Import and apply architecture optimizations early (before any Triton compilation)
# This ensures the Triton SM architecture patch is applied before benchmarks run
try:
    import arch_config  # noqa: F401 - triggers configure_optimizations() at module level
except ImportError:
    pass  # If arch_config not available, continue without optimizations

# Import benchmark testing functionality
try:
    import torch
    from common.python.chapter_compare_template import discover_benchmarks, load_benchmark
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


def ensure_peak_benchmarks_exist():
    """Ensure peak benchmark results exist, run if missing."""
    # Check for peak benchmark results
    peak_files = list(repo_root.glob("benchmark_peak_results_*.json"))
    if not peak_files:
        # Also check uppercase pattern for backwards compatibility
        peak_files = list(repo_root.glob("BENCHMARK_PEAK_RESULTS_*.json"))
    
    if peak_files:
        # Results exist, use them
        return
    
    # No results found - need to run peak benchmarks
    print("\n" + "=" * 80)
    print("Peak Performance Benchmark Results Not Found")
    print("=" * 80)
    print("Running peak performance detection...")
    print()
    
    try:
        import subprocess
        import sys
        
        # Run benchmark_peak.py
        benchmark_peak_script = repo_root / "tools" / "benchmarking" / "benchmark_peak.py"
        if benchmark_peak_script.exists():
            result = subprocess.run(
                [sys.executable, str(benchmark_peak_script), "--output-dir", str(repo_root)],
                cwd=str(repo_root),
                capture_output=False,
                timeout=15  # 15 second timeout to prevent hangs
            )
            if result.returncode == 0:
                print("\nPeak performance benchmarks completed successfully")
            else:
                print("\nWARNING: Peak performance benchmarks had issues, but continuing...")
        else:
            print("WARNING: benchmark_peak.py not found, skipping peak detection")
    except subprocess.TimeoutExpired:
        print("\nWARNING: Peak performance benchmarks timed out, but continuing...")
    except Exception as e:
        print(f"\nWARNING: Could not run peak benchmarks: {e}")
        print("   Continuing with benchmarks (using hardcoded targets if available)...")


def run_benchmarks(chapter='all', format='both', enable_profiling=False, suite_timeout_seconds=None):
    """Run all benchmarks - discover, run, and summarize results.
    
    THIS IS THE DEFAULT ACTION WHEN RUNNING benchmark.py
    
    Args:
        chapter: Chapter to run ('all' or specific chapter like 'ch1')
        format: Output format ('json', 'markdown', or 'both')
        enable_profiling: If True, generate nsys-rep files alongside benchmarks
        suite_timeout_seconds: Overall timeout for entire suite in seconds (default: 14400 = 4 hours)
                              Set to None to disable timeout
    """
    dump_environment_and_capabilities()

    if not BENCHMARK_AVAILABLE:
        print("FAILED: Benchmark testing requires torch and benchmark_harness")
        print("   Install dependencies: pip install -r requirements_latest.txt")
        sys.exit(1)
    
    # Ensure peak benchmarks exist before running tests
    if torch.cuda.is_available():
        ensure_peak_benchmarks_exist()
    
    # Import test functionality
    from tools.testing.run_all_benchmarks import test_chapter, generate_markdown_report
    
    print("=" * 80)
    print("RUNNING ALL BENCHMARKS")
    if enable_profiling:
        print("PROFILING ENABLED: nsys-rep files will be generated")
    
    # Set realistic default timeout: ~4 hours (14400 seconds)
    # With 223 benchmark pairs × 2 benchmarks × 15s timeout = ~111 minutes minimum
    # Plus overhead for discovery, setup, teardown, compilation, etc. = ~2-3 hours total
    # 4 hours provides comfortable margin for profiling, slower systems, etc.
    if suite_timeout_seconds is None:
        suite_timeout_seconds = 14400  # 4 hours default
    
    if suite_timeout_seconds > 0:
        timeout_hours = suite_timeout_seconds / 3600
        print(f"SUITE TIMEOUT: {timeout_hours:.1f} hours ({suite_timeout_seconds} seconds)")
        print("  (Individual benchmarks have 15s timeout to prevent hangs)")
    else:
        print("SUITE TIMEOUT: Disabled (will run until completion)")
    print("=" * 80)
    print()
    
    # Determine chapters to test
    if chapter and chapter != 'all':
        # Normalize chapter argument: accept both "12" and "ch12"
        if chapter.isdigit():
            chapter = f"ch{chapter}"
        elif chapter.startswith('ch') and chapter[2:].isdigit():
            pass  # Already in correct format
        # Use the normalized chapter name
        chapter_dirs = [repo_root / chapter]
    else:
        chapter_dirs = sorted([
            d for d in repo_root.iterdir()
            if d.is_dir() and d.name.startswith('ch') and d.name[2:].isdigit()
        ])
    
    # Test all chapters with suite-level timeout protection
    import signal
    import time
    
    start_time = time.time()
    all_results = []
    suite_timed_out = False
    
    def timeout_handler(signum, frame):
        nonlocal suite_timed_out
        suite_timed_out = True
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"SUITE TIMEOUT: Benchmark suite exceeded {suite_timeout_seconds}s timeout")
        print(f"  Elapsed time: {elapsed/3600:.2f} hours")
        print(f"  Chapters completed: {len(all_results)}")
        print(f"{'='*80}\n")
        raise TimeoutError(f"Suite timeout after {suite_timeout_seconds} seconds")
    
    # Set up timeout signal handler if timeout is enabled
    if suite_timeout_seconds > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(suite_timeout_seconds)
    
    try:
        for chapter_dir in chapter_dirs:
            if not chapter_dir.exists():
                continue
            
            # Check if we've exceeded timeout
            if suite_timeout_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed >= suite_timeout_seconds:
                    print(f"\nWARNING: Approaching suite timeout, skipping remaining chapters")
                    break
            
            result = test_chapter(chapter_dir, enable_profiling=enable_profiling)
            all_results.append(result)
    except (TimeoutError, KeyboardInterrupt):
        suite_timed_out = True
        print("\nBenchmark suite interrupted")
    finally:
        # Cancel timeout alarm
        if suite_timeout_seconds > 0:
            signal.alarm(0)
    
    # Save results
    output_json = repo_root / 'benchmark_test_results.json'
    output_md = repo_root / 'benchmark_test_results.md'
    
    if format in ['json', 'both']:
        with open(output_json, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
            }, f, indent=2)
        print(f"\nPASSED: JSON results saved to: {output_json}")
    
    if format in ['markdown', 'both']:
        generate_markdown_report(all_results, output_md)
        print(f"PASSED: Markdown report saved to: {output_md}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_benchmarks = sum(r['summary']['total_benchmarks'] for r in all_results)
    total_successful = sum(r['summary']['successful'] for r in all_results)
    total_failed = sum(r['summary']['failed'] for r in all_results)
    
    print(f"Total benchmarks tested: {total_benchmarks}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    
    if total_benchmarks > 0:
        success_rate = (total_successful / total_benchmarks) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
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
    
    if total_failed > 0:
        sys.exit(1)
    
    # Print profiling summary if profiling was enabled
    if enable_profiling:
        profiling_dir = repo_root / "benchmark_profiles"
        if profiling_dir.exists():
            # Count nsys-rep files
            nsys_files = list(profiling_dir.rglob("*.nsys-rep"))
            if nsys_files:
                print("\n" + "=" * 80)
                print("PROFILING SUMMARY")
                print("=" * 80)
                print(f"nsys-rep files generated: {len(nsys_files)}")
                print(f"Profiling output directory: {profiling_dir}")
                print("\nTo view profiles:")
                print(f"  nsys-ui {profiling_dir}/ch*/<benchmark_name>_*.nsys-rep")
                print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks - discover, run, and summarize results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--chapter', type=str, default='all', help='Chapter to run (e.g., 12, ch12, or "all") (default: all)')
    parser.add_argument('--format', choices=['json', 'markdown', 'both'], default='both', help='Output format (default: both)')
    parser.add_argument('--profile', action='store_true', help='Enable nsys profiling (generates .nsys-rep files alongside benchmarks)')
    parser.add_argument('--suite-timeout', type=int, default=14400, 
                       help='Overall timeout for entire suite in seconds (default: 14400 = 4 hours). Set to 0 to disable.')
    
    args = parser.parse_args()
    
    # Run benchmarks by default
    suite_timeout = args.suite_timeout if args.suite_timeout > 0 else None
    run_benchmarks(chapter=args.chapter, format=args.format, enable_profiling=args.profile, 
                   suite_timeout_seconds=suite_timeout)


if __name__ == '__main__':
    main()

