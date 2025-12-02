#!/usr/bin/env python3
"""Enhanced benchmark runner with comprehensive metric tracking.

This script:
1. Runs all baseline/optimized pairs with full profiling (nsys/ncu/torch)
2. Tracks if benchmarks improve over baseline
3. Tracks if metrics are successfully collected
4. Generates a comprehensive status table
5. Identifies benchmarks that need fixing

Usage:
    python core/harness/run_benchmarks_with_metrics.py [--targets chX chY:example]
"""

import sys
from pathlib import Path
import json
import argparse
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass

# Ensure repository root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.env import apply_env_defaults, dump_environment_and_capabilities
apply_env_defaults()

import torch
from core.harness.run_benchmarks import (
    discover_benchmarks, discover_cuda_benchmarks,
    load_benchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig,
    profile_python_benchmark, profile_cuda_executable,
    find_cuda_executable, benchmark_cuda_executable,
    ensure_cuda_executables_built, reset_cuda_state,
    format_time_ms
)
from core.discovery import chapter_slug, resolve_target_chapters


@dataclass
class MetricCollectionStatus:
    """Status of metric collection for a benchmark."""
    nsys: bool = False  # nsys-rep file exists and is valid
    ncu: bool = False   # ncu-rep file exists and is valid
    torch: bool = False  # torch profiler data collected
    nsys_path: Optional[Path] = None
    ncu_path: Optional[Path] = None
    torch_path: Optional[Path] = None


@dataclass
class BenchmarkStatus:
    """Comprehensive status for a benchmark pair."""
    chapter: str
    example: str
    baseline_file: str
    optimized_file: str
    type: str  # 'python' or 'cuda'
    
    # Execution status
    baseline_executed: bool = False
    optimized_executed: bool = False
    baseline_failed: bool = False
    optimized_failed: bool = False
    error_msg: Optional[str] = None
    
    # Performance metrics
    baseline_time_ms: Optional[float] = None
    optimized_time_ms: Optional[float] = None
    speedup: float = 1.0
    improves: bool = False  # speedup > 1.0
    
    # Metric collection
    baseline_metrics: MetricCollectionStatus = None
    optimized_metrics: MetricCollectionStatus = None
    
    def __post_init__(self):
        if self.baseline_metrics is None:
            self.baseline_metrics = MetricCollectionStatus()
        if self.optimized_metrics is None:
            self.optimized_metrics = MetricCollectionStatus()
    
    @property
    def all_criteria_met(self) -> bool:
        """Check if all success criteria are met."""
        return (
            self.baseline_executed and
            self.optimized_executed and
            not self.baseline_failed and
            not self.optimized_failed and
            self.improves and
            self.baseline_metrics.nsys and
            self.baseline_metrics.ncu and
            self.optimized_metrics.nsys and
            self.optimized_metrics.ncu
        )
    
    @property
    def issues(self) -> List[str]:
        """List of issues preventing success."""
        issues = []
        if not self.baseline_executed:
            issues.append("Baseline not executed")
        if not self.optimized_executed:
            issues.append("Optimized not executed")
        if self.baseline_failed:
            issues.append(f"Baseline failed: {self.error_msg}")
        if self.optimized_failed:
            issues.append(f"Optimized failed: {self.error_msg}")
        if not self.improves:
            issues.append(f"No improvement (speedup: {self.speedup:.2f}x)")
        if not self.baseline_metrics.nsys:
            issues.append("Baseline nsys metrics missing")
        if not self.baseline_metrics.ncu:
            issues.append("Baseline ncu metrics missing")
        if not self.optimized_metrics.nsys:
            issues.append("Optimized nsys metrics missing")
        if not self.optimized_metrics.ncu:
            issues.append("Optimized ncu metrics missing")
        return issues


def check_ncu_available() -> bool:
    """Check if ncu is available."""
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


def check_nsys_available() -> bool:
    """Check if nsys is available."""
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


def profile_with_ncu_python(
    benchmark: Any,
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> Optional[Path]:
    """Profile a Python benchmark using ncu."""
    if not check_ncu_available():
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_name = benchmark_path.stem
    ncu_output = output_dir / f"{benchmark_name}_{variant}.ncu-rep"
    
    # Check if file already exists (from previous run)
    if ncu_output.exists():
        return ncu_output
    # Check for any matching .ncu-rep file
    for existing_file in output_dir.glob(f"{benchmark_name}_{variant}*.ncu-rep"):
        return existing_file
    
    wrapper_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    
    try:
        wrapper_script.write(f"""
import sys
from pathlib import Path

sys.path.insert(0, r'{chapter_dir}')

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
        
        ncu_command = [
            "ncu",
            "--set", "full",
            "--metrics", "gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active",
            "--replay-mode", "kernel",
            "-o", str(ncu_output.with_suffix("")),
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
        
        Path(wrapper_script.name).unlink(missing_ok=True)
        
        # Check if file exists (NCU may create file even with non-zero exit code)
        # Also check with .ncu-rep extension
        if ncu_output.exists():
            return ncu_output
        # Try alternative path (NCU might add extension differently)
        alt_path = output_dir / f"{benchmark_name}_{variant}.ncu-rep"
        if alt_path.exists():
            return alt_path
        # Check for any .ncu-rep file matching the pattern
        for ncu_file in output_dir.glob(f"{benchmark_name}_{variant}*.ncu-rep"):
            return ncu_file
        return None
    except (subprocess.SubprocessError, OSError) as e:
        Path(wrapper_script.name).unlink(missing_ok=True)
        print(f"NCU profiling failed: {e}")
        return None


def profile_with_ncu_cuda(
    executable: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> Optional[Path]:
    """Profile a CUDA executable using ncu."""
    if not check_ncu_available():
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    exec_name = executable.stem
    ncu_output = output_dir / f"{exec_name}_{variant}.ncu-rep"
    
    # Check if file already exists (from previous run)
    if ncu_output.exists():
        return ncu_output
    # Check for any matching .ncu-rep file
    for existing_file in output_dir.glob(f"{exec_name}_{variant}*.ncu-rep"):
        return existing_file
    
    ncu_command = [
        "ncu",
        "--set", "full",
        "--metrics", "gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active",
        "--replay-mode", "kernel",
        "-o", str(ncu_output.with_suffix("")),
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
        
        # Check if file exists (NCU may create file even with non-zero exit code)
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
    except Exception:
        return None


def collect_torch_profiler_metrics(
    benchmark: Any,
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline"
) -> bool:
    """Collect PyTorch profiler metrics (just check if available, don't store full data)."""
    # For now, we'll use torch.profiler availability as the check
    # Full implementation would wrap benchmark execution with profiler
    try:
        from torch.profiler import profile as torch_profile, ProfilerActivity
        return True
    except ImportError:
        return False


def test_benchmark_pair_with_metrics(
    chapter_dir: Path,
    baseline_path: Path,
    optimized_path: Path,
    example_name: str,
    profiling_output_dir: Path,
    harness: BenchmarkHarness
) -> BenchmarkStatus:
    """Test a benchmark pair with full metric collection."""
    chapter_name = chapter_slug(chapter_dir, repo_root)
    status = BenchmarkStatus(
        chapter=chapter_name,
        example=example_name,
        baseline_file=baseline_path.name,
        optimized_file=optimized_path.name,
        type='python'
    )
    
    # Load and run baseline
    reset_cuda_state()
    baseline_benchmark = load_benchmark(baseline_path)
    if baseline_benchmark is None:
        status.baseline_failed = True
        status.error_msg = "Failed to load baseline"
        return status
    
    try:
        baseline_result = harness.benchmark(baseline_benchmark)
        status.baseline_executed = True
        status.baseline_time_ms = baseline_result.timing.mean_ms if baseline_result.timing else None
        
        # Profile baseline with nsys
        nsys_path = profile_python_benchmark(
            baseline_benchmark, baseline_path, chapter_dir,
            profiling_output_dir, variant="baseline"
        )
        status.baseline_metrics.nsys = nsys_path is not None
        status.baseline_metrics.nsys_path = nsys_path
        
        # Profile baseline with ncu
        ncu_path = profile_with_ncu_python(
            baseline_benchmark, baseline_path, chapter_dir,
            profiling_output_dir, variant="baseline"
        )
        status.baseline_metrics.ncu = ncu_path is not None
        status.baseline_metrics.ncu_path = ncu_path
        
        # Check torch profiler availability
        status.baseline_metrics.torch = collect_torch_profiler_metrics(
            baseline_benchmark, baseline_path, chapter_dir,
            profiling_output_dir, variant="baseline"
        )
        
    except Exception as e:
        status.baseline_failed = True
        status.error_msg = f"Baseline failed: {str(e)}"
        return status
    
    # Load and run optimized
    reset_cuda_state()
    optimized_benchmark = load_benchmark(optimized_path)
    if optimized_benchmark is None:
        status.optimized_failed = True
        status.error_msg = "Failed to load optimized"
        return status
    
    try:
        optimized_result = harness.benchmark(optimized_benchmark)
        status.optimized_executed = True
        status.optimized_time_ms = optimized_result.timing.mean_ms if optimized_result.timing else None
        
        if status.baseline_time_ms and status.optimized_time_ms > 0:
            status.speedup = status.baseline_time_ms / status.optimized_time_ms
            status.improves = status.speedup > 1.0
        
        # Profile optimized with nsys
        nsys_path = profile_python_benchmark(
            optimized_benchmark, optimized_path, chapter_dir,
            profiling_output_dir, variant="optimized"
        )
        status.optimized_metrics.nsys = nsys_path is not None
        status.optimized_metrics.nsys_path = nsys_path
        
        # Profile optimized with ncu
        ncu_path = profile_with_ncu_python(
            optimized_benchmark, optimized_path, chapter_dir,
            profiling_output_dir, variant="optimized"
        )
        status.optimized_metrics.ncu = ncu_path is not None
        status.optimized_metrics.ncu_path = ncu_path
        
        # Check torch profiler availability
        status.optimized_metrics.torch = collect_torch_profiler_metrics(
            optimized_benchmark, optimized_path, chapter_dir,
            profiling_output_dir, variant="optimized"
        )
        
    except Exception as e:
        status.optimized_failed = True
        status.error_msg = f"Optimized failed: {str(e)}"
        return status
    
    return status


def test_cuda_pair_with_metrics(
    chapter_dir: Path,
    baseline_cu_path: Path,
    optimized_cu_path: Path,
    example_name: str,
    profiling_output_dir: Path
) -> BenchmarkStatus:
    """Test a CUDA benchmark pair with full metric collection."""
    chapter_name = chapter_slug(chapter_dir, repo_root)
    status = BenchmarkStatus(
        chapter=chapter_name,
        example=example_name,
        baseline_file=baseline_cu_path.name,
        optimized_file=optimized_cu_path.name,
        type='cuda'
    )
    
    # Find baseline executable
    baseline_executable = find_cuda_executable(baseline_cu_path, chapter_dir)
    if baseline_executable is None:
        status.baseline_failed = True
        status.error_msg = f"Baseline executable not found"
        return status
    
    # Benchmark baseline
    baseline_result = benchmark_cuda_executable(baseline_executable, iterations=20, warmup=5, timeout=15)
    if baseline_result is None:
        status.baseline_failed = True
        status.error_msg = "Baseline execution failed or timed out"
        return status
    if baseline_result.skip_reason:
        status.baseline_failed = True
        status.error_msg = baseline_result.skip_reason
        return status
    
    status.baseline_executed = True
    status.baseline_time_ms = baseline_result.mean_ms
    
    # Profile baseline
    nsys_path = profile_cuda_executable(
        baseline_executable, chapter_dir, profiling_output_dir, variant="baseline"
    )
    status.baseline_metrics.nsys = nsys_path is not None
    status.baseline_metrics.nsys_path = nsys_path
    
    ncu_path = profile_with_ncu_cuda(
        baseline_executable, chapter_dir, profiling_output_dir, variant="baseline"
    )
    status.baseline_metrics.ncu = ncu_path is not None
    status.baseline_metrics.ncu_path = ncu_path
    
    # Find optimized executable
    optimized_executable = find_cuda_executable(optimized_cu_path, chapter_dir)
    if optimized_executable is None:
        status.optimized_failed = True
        status.error_msg = "Optimized executable not found"
        return status
    
    # Benchmark optimized
    optimized_result = benchmark_cuda_executable(optimized_executable, iterations=20, warmup=5, timeout=15)
    if optimized_result is None:
        status.optimized_failed = True
        status.error_msg = "Optimized execution failed or timed out"
        return status
    if optimized_result.skip_reason:
        status.optimized_failed = True
        status.error_msg = optimized_result.skip_reason
        return status
    
    status.optimized_executed = True
    status.optimized_time_ms = optimized_result.mean_ms
    
    if status.baseline_time_ms and status.optimized_time_ms > 0:
        status.speedup = status.baseline_time_ms / status.optimized_time_ms
        status.improves = status.speedup > 1.0
    
    # Profile optimized
    nsys_path = profile_cuda_executable(
        optimized_executable, chapter_dir, profiling_output_dir, variant="optimized"
    )
    status.optimized_metrics.nsys = nsys_path is not None
    status.optimized_metrics.nsys_path = nsys_path
    
    ncu_path = profile_with_ncu_cuda(
        optimized_executable, chapter_dir, profiling_output_dir, variant="optimized"
    )
    status.optimized_metrics.ncu = ncu_path is not None
    status.optimized_metrics.ncu_path = ncu_path
    
    return status


def test_chapter_with_metrics(chapter_dir: Path, example_filters: Optional[Set[str]] = None) -> List[BenchmarkStatus]:
    """Test benchmarks in a chapter with full metric tracking."""
    dump_environment_and_capabilities()
    
    chapter_id = chapter_slug(chapter_dir, repo_root)
    print(f"\n{'='*80}")
    print(f"Testing {chapter_id.upper()} with Full Metrics")
    print(f"{'='*80}")
    
    if not torch.cuda.is_available():
        print(f"  CUDA not available - skipping")
        return []
    
    reset_cuda_state()
    
    # Set up profiling output directory
    profiling_output_dir = repo_root / "benchmark_profiles" / Path(chapter_id)
    if check_nsys_available() or check_ncu_available():
        profiling_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Profiling enabled: profiles saved to {profiling_output_dir}")
    
    # Create harness
    config = BenchmarkConfig(
        iterations=20,
        warmup=5,
        timeout_seconds=15,
        enable_memory_tracking=True
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    all_statuses = []
    
    # Discover Python benchmarks
    print(f"  Discovering Python benchmarks...", flush=True)
    python_pairs = discover_benchmarks(chapter_dir)
    if example_filters:
        python_pairs = [pair for pair in python_pairs if pair[2] in example_filters]
        print(f"  Filtered to {len(python_pairs)} Python benchmark pair(s)")
    else:
        print(f"  Found {len(python_pairs)} Python benchmark pair(s)")
    
    # Discover CUDA benchmarks
    print(f"  Discovering CUDA benchmarks...", flush=True)
    cuda_pairs = discover_cuda_benchmarks(chapter_dir)
    if example_filters:
        cuda_pairs = [pair for pair in cuda_pairs if pair[2] in example_filters]
    if cuda_pairs:
        print(f"  Found {len(cuda_pairs)} CUDA benchmark pair(s)")
        ensure_cuda_executables_built(chapter_dir)
    
    # Test Python pairs
    for baseline_path, optimized_paths, example_name in python_pairs:
        for optimized_path in optimized_paths:
            print(f"\n  Testing: {example_name}")
            print(f"    Baseline: {baseline_path.name}")
            print(f"    Optimized: {optimized_path.name}")
            
            status = test_benchmark_pair_with_metrics(
                chapter_dir, baseline_path, optimized_path, example_name,
                profiling_output_dir, harness
            )
            
            all_statuses.append(status)
            
            # Print quick status
            if status.all_criteria_met:
                print(f"    ✓ PASS: All criteria met")
            else:
                print(f"    ✗ ISSUES: {', '.join(status.issues)}")
            
            reset_cuda_state()
    
    # Test CUDA pairs
    for baseline_cu_path, optimized_cu_paths, example_name in cuda_pairs:
        for optimized_cu_path in optimized_cu_paths:
            print(f"\n  Testing (CUDA): {example_name}")
            print(f"    Baseline: {baseline_cu_path.name}")
            print(f"    Optimized: {optimized_cu_path.name}")
            
            status = test_cuda_pair_with_metrics(
                chapter_dir, baseline_cu_path, optimized_cu_path, example_name,
                profiling_output_dir
            )
            
            all_statuses.append(status)
            
            # Print quick status
            if status.all_criteria_met:
                print(f"    ✓ PASS: All criteria met")
            else:
                print(f"    ✗ ISSUES: {', '.join(status.issues)}")
            
            reset_cuda_state()
    
    return all_statuses


def generate_status_table(all_statuses: List[BenchmarkStatus]) -> str:
    """Generate a comprehensive status table."""
    lines = []
    lines.append("=" * 150)
    lines.append("COMPREHENSIVE BENCHMARK STATUS TABLE")
    lines.append("=" * 150)
    lines.append("")
    
    # Header
    header = (
        f"{'Chapter':<10} {'Example':<25} {'Type':<8} {'Status':<12} "
        f"{'Speedup':<10} {'Improv':<8} {'B-nsys':<8} {'B-ncu':<8} "
        f"{'O-nsys':<8} {'O-ncu':<8} {'Issues'}"
    )
    lines.append(header)
    lines.append("-" * 150)
    
    # Rows
    for status in sorted(all_statuses, key=lambda s: (s.chapter, s.example)):
        status_str = "PASS" if status.all_criteria_met else "FAIL"
        speedup_str = f"{status.speedup:.2f}x" if status.speedup else "N/A"
        improv_str = "✓" if status.improves else "✗"
        b_nsys = "✓" if status.baseline_metrics.nsys else "✗"
        b_ncu = "✓" if status.baseline_metrics.ncu else "✗"
        o_nsys = "✓" if status.optimized_metrics.nsys else "✗"
        o_ncu = "✓" if status.optimized_metrics.ncu else "✗"
        
        issues_str = "; ".join(status.issues) if status.issues else "None"
        if len(issues_str) > 50:
            issues_str = issues_str[:47] + "..."
        
        row = (
            f"{status.chapter:<10} {status.example:<25} {status.type:<8} "
            f"{status_str:<12} {speedup_str:<10} {improv_str:<8} "
            f"{b_nsys:<8} {b_ncu:<8} {o_nsys:<8} {o_ncu:<8} {issues_str}"
        )
        lines.append(row)
    
    lines.append("-" * 150)
    
    # Summary
    total = len(all_statuses)
    passed = sum(1 for s in all_statuses if s.all_criteria_met)
    failed = total - passed
    
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"  Total benchmarks: {total}")
    lines.append(f"  Passed (all criteria met): {passed}")
    lines.append(f"  Failed (need fixing): {failed}")
    lines.append("")
    
    # Breakdown by issue type
    no_improve = sum(1 for s in all_statuses if s.improves == False and s.baseline_executed and s.optimized_executed)
    missing_nsys = sum(1 for s in all_statuses if not s.baseline_metrics.nsys or not s.optimized_metrics.nsys)
    missing_ncu = sum(1 for s in all_statuses if not s.baseline_metrics.ncu or not s.optimized_metrics.ncu)
    exec_failures = sum(1 for s in all_statuses if s.baseline_failed or s.optimized_failed)
    
    lines.append("BREAKDOWN BY ISSUE TYPE:")
    lines.append(f"  No improvement: {no_improve}")
    lines.append(f"  Missing nsys metrics: {missing_nsys}")
    lines.append(f"  Missing ncu metrics: {missing_ncu}")
    lines.append(f"  Execution failures: {exec_failures}")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Run all benchmarks with comprehensive metric tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        help=("Space-separated list of targets. "
              "Use 'ch03' to test an entire chapter or 'ch03:resnet_50' "
              "to run baseline_resnet_50 and optimized_resnet_50. "
              "Omit this flag (or pass 'all') to run every chapter.")
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=repo_root / 'benchmark_status_table.txt',
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RUNNING ALL BENCHMARKS WITH COMPREHENSIVE METRICS")
    print("=" * 80)
    print()
    
    dump_environment_and_capabilities()
    print()
    
    # Check profiler availability
    print("Checking profiler availability...")
    nsys_avail = check_nsys_available()
    ncu_avail = check_ncu_available()
    print(f"  nsys available: {nsys_avail}")
    print(f"  ncu available: {ncu_avail}")
    print()
    
    # Determine chapters to test
    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(args.targets)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    
    # Test all chapters
    all_statuses = []
    for chapter_dir in chapter_dirs:
        if not chapter_dir.exists():
            continue
        
        example_filters = chapter_filters.get(chapter_slug(chapter_dir, repo_root))
        statuses = test_chapter_with_metrics(chapter_dir, example_filters=example_filters)
        all_statuses.extend(statuses)
    
    # Generate table
    table = generate_status_table(all_statuses)
    print("\n" + table)
    
    # Save to file
    args.output.write_text(table)
    print(f"\nStatus table saved to: {args.output}")
    
    # Save JSON for programmatic access
    json_output = args.output.with_suffix('.json')
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'total': len(all_statuses),
        'passed': sum(1 for s in all_statuses if s.all_criteria_met),
        'failed': sum(1 for s in all_statuses if not s.all_criteria_met),
        'statuses': [
            {
                'chapter': s.chapter,
                'example': s.example,
                'baseline_file': s.baseline_file,
                'optimized_file': s.optimized_file,
                'type': s.type,
                'baseline_executed': s.baseline_executed,
                'optimized_executed': s.optimized_executed,
                'baseline_failed': s.baseline_failed,
                'optimized_failed': s.optimized_failed,
                'error_msg': s.error_msg,
                'baseline_time_ms': s.baseline_time_ms,
                'optimized_time_ms': s.optimized_time_ms,
                'speedup': s.speedup,
                'improves': s.improves,
                'baseline_metrics': {
                    'nsys': s.baseline_metrics.nsys,
                    'ncu': s.baseline_metrics.ncu,
                    'torch': s.baseline_metrics.torch,
                    'nsys_path': str(s.baseline_metrics.nsys_path) if s.baseline_metrics.nsys_path else None,
                    'ncu_path': str(s.baseline_metrics.ncu_path) if s.baseline_metrics.ncu_path else None,
                },
                'optimized_metrics': {
                    'nsys': s.optimized_metrics.nsys,
                    'ncu': s.optimized_metrics.ncu,
                    'torch': s.optimized_metrics.torch,
                    'nsys_path': str(s.optimized_metrics.nsys_path) if s.optimized_metrics.nsys_path else None,
                    'ncu_path': str(s.optimized_metrics.ncu_path) if s.optimized_metrics.ncu_path else None,
                },
                'all_criteria_met': s.all_criteria_met,
                'issues': s.issues,
            }
            for s in all_statuses
        ]
    }
    json_output.write_text(json.dumps(json_data, indent=2))
    print(f"JSON data saved to: {json_output}")
    
    return 0 if all(s.all_criteria_met for s in all_statuses) else 1


if __name__ == '__main__':
    sys.exit(main())
