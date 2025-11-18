"""Unified CLI for benchmark execution and management using Typer."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

try:
    import typer
    from typer import Option
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False
    typer = None
    Option = None
    Argument = None
    Context = None

# Suppress CUDA capability warnings
warnings.filterwarnings("ignore", message=".*Found GPU.*which is of cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

def _expand_multi_value_option(option_names: List[str]) -> None:
    """Allow passing `--option value1 value2` by rewriting argv."""
    argv = sys.argv
    if not set(option_names).intersection(argv):
        return
    new_argv = [argv[0]]
    i = 1
    option_set = set(option_names)
    while i < len(argv):
        token = argv[i]
        if token in option_set:
            option = token
            i += 1
            consumed = False
            while i < len(argv) and not argv[i].startswith("-"):
                new_argv.append(option)
                new_argv.append(argv[i])
                i += 1
                consumed = True
            if not consumed:
                new_argv.append(option)
            continue
        new_argv.append(token)
        i += 1
    sys.argv = new_argv


_expand_multi_value_option(["--targets", "-t"])

from common.python.env_defaults import apply_env_defaults, dump_environment_and_capabilities
from common.python.logger import setup_logging, get_logger, log_benchmark_start, log_benchmark_complete, log_benchmark_error
from common.python.artifact_manager import ArtifactManager
from tools.verification.verify_all_benchmarks import resolve_target_chapters, run_verification
from common.python import profiler_config as profiler_config_mod
from common.python.discovery import chapter_slug, discover_all_chapters

apply_env_defaults()

def _validate_ncu_metric_set(metric_set: str) -> str:
    normalized = metric_set.strip().lower()
    valid = {"auto", "deep_dive", "roofline", "minimal"}
    if normalized not in valid:
        message = (
            f"Invalid Nsight Compute metric set '{metric_set}'. "
            "Choose from 'auto', 'deep_dive', 'roofline', or 'minimal'."
        )
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    if normalized != "auto":
        profiler_config_mod.set_default_profiler_metric_set(normalized)
    return normalized


def _validate_profile_type(profile: str) -> str:
    normalized = profile.strip().lower()
    valid = {"none", "minimal", "deep_dive", "roofline"}
    if normalized not in valid:
        message = (
            f"Invalid profile choice '{profile}'. "
            "Choose from 'none', 'minimal', 'deep_dive', or 'roofline'."
        )
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


def _parse_target_extra_args(entries: Optional[List[str]]) -> Dict[str, List[str]]:
    """Parse --target-extra-arg entries of the form target=\"--flag value\"."""
    parsed: Dict[str, List[str]] = {}
    for entry in entries or []:
        target, sep, args = entry.partition("=")
        if not sep or not target or not args:
            continue
        parsed[target.strip()] = shlex.split(args)
    return parsed

# Import architecture optimizations early
try:
    import arch_config  # noqa: F401
except ImportError:
    pass

# Import benchmark functionality
try:
    import torch
    from common.python.chapter_compare_template import discover_benchmarks, load_benchmark
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Import test functionality
try:
    from tools.testing.run_all_benchmarks import test_chapter, generate_markdown_report
    TEST_FUNCTIONS_AVAILABLE = True
except ImportError:
    TEST_FUNCTIONS_AVAILABLE = False

if TYPER_AVAILABLE:
    app = typer.Typer(
        name="benchmark",
        help="Unified benchmark execution and management CLI",
        add_completion=False,
    )
else:
    app = None


def ensure_peak_benchmarks_exist():
    """Ensure peak benchmark results exist, run if missing."""
    peak_files = list(repo_root.glob("benchmark_peak_results_*.json"))
    if not peak_files:
        peak_files = list(repo_root.glob("BENCHMARK_PEAK_RESULTS_*.json"))
    
    if peak_files:
        return
    
    logger = get_logger(__name__)
    logger.info("Peak performance benchmark results not found, running detection...")
    
    try:
        import subprocess
        benchmark_peak_script = repo_root / "tools" / "benchmarking" / "benchmark_peak.py"
        if benchmark_peak_script.exists():
            result = subprocess.run(
                [sys.executable, str(benchmark_peak_script), "--output-dir", str(repo_root)],
                cwd=str(repo_root),
                capture_output=False,
                timeout=90  # Peak detection takes 30-60 seconds per README, use 90s for safety
            )
            if result.returncode == 0:
                logger.info("Peak performance benchmarks completed successfully")
            else:
                logger.warning("Peak performance benchmarks had issues, but continuing...")
        else:
            logger.warning("benchmark_peak.py not found, skipping peak detection")
    except subprocess.TimeoutExpired:
        logger.warning("Peak performance benchmarks timed out, but continuing...")
    except Exception as e:
        logger.warning(f"Could not run peak benchmarks: {e}, continuing...")


def _execute_benchmarks(
    targets: Optional[List[str]] = None,
    output_format: str = "both",
    profile_type: str = "none",
    suite_timeout: Optional[int] = None,
    timeout_multiplier: float = 1.0,
    reproducible: bool = False,
    cold_start: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    force_pipeline: bool = False,
    artifacts_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    ncu_metric_set: str = "auto",
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    torchrun_env: Optional[List[str]] = None,
    target_extra_args: Optional[List[str]] = None,
):
    _execute_benchmarks_impl(
        targets=targets,
        output_format=output_format,
        profile_type=profile_type,
        suite_timeout=suite_timeout,
        timeout_multiplier=timeout_multiplier,
        reproducible=reproducible,
        cold_start=cold_start,
        iterations=iterations,
        warmup=warmup,
        force_pipeline=force_pipeline,
        artifacts_dir=artifacts_dir,
        log_level=log_level,
        log_file=log_file,
        accept_regressions=accept_regressions,
        update_expectations=update_expectations,
        ncu_metric_set=ncu_metric_set,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        torchrun_env=torchrun_env,
        target_extra_args=target_extra_args,
    )


def _execute_benchmarks_impl(
    targets: Optional[List[str]] = None,
    output_format: str = "both",
    profile_type: str = "none",
    suite_timeout: Optional[int] = None,
    timeout_multiplier: float = 1.0,
    reproducible: bool = False,
    cold_start: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    force_pipeline: bool = False,
    artifacts_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    ncu_metric_set: str = "auto",
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    torchrun_env: Optional[List[str]] = None,
    target_extra_args: Optional[List[str]] = None,
):
    """Shared function to execute benchmarks."""
    parsed_extra_args = _parse_target_extra_args(target_extra_args)
    # Set force pipeline flag before benchmarks run
    from common.python.cuda_capabilities import set_force_pipeline
    set_force_pipeline(force_pipeline)
    
    # Setup logging
    artifact_manager = ArtifactManager(base_dir=Path(artifacts_dir) if artifacts_dir else None)
    
    if log_file is None:
        log_file = artifact_manager.get_log_path()
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        log_format="json",
        use_rich=True
    )
    
    logger = get_logger(__name__)
    
    if not BENCHMARK_AVAILABLE:
        logger.error("Benchmark testing requires torch and benchmark_harness")
        logger.error("Install dependencies: pip install -r requirements_latest.txt")
        sys.exit(1)
    
    dump_environment_and_capabilities()
    
    # Ensure peak benchmarks exist
    if torch.cuda.is_available():
        ensure_peak_benchmarks_exist()
    
    if not TEST_FUNCTIONS_AVAILABLE:
        logger.error("Test functions not available - cannot run benchmarks")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("RUNNING ALL BENCHMARKS")
    profile_choice = (profile_type or "none").lower()
    enable_profiling = profile_choice != "none"
    if enable_profiling:
        logger.info(f"PROFILING ENABLED ({profile_choice}): nsys/ncu/PyTorch profiling will run")
    else:
        logger.info("PROFILING DISABLED (profile=none)")
    if timeout_multiplier != 1.0:
        logger.info(f"TIMEOUT MULTIPLIER: {timeout_multiplier}x (all timeouts scaled by this factor)")
    if reproducible:
        logger.info("REPRODUCIBLE MODE: All seeds set to 42, deterministic algorithms enabled; slower fallback kernels and ops lacking deterministic support may raise.")
    if cold_start:
        logger.info("COLD START MODE: GPU state will be reset between benchmarks")
    
    # Set default timeout
    if suite_timeout is None:
        suite_timeout = 14400  # 4 hours
    
    if suite_timeout > 0:
        timeout_hours = suite_timeout / 3600
        logger.info(f"SUITE TIMEOUT: {timeout_hours:.1f} hours ({suite_timeout} seconds)")
    else:
        logger.info("SUITE TIMEOUT: Disabled")
    logger.info("=" * 80)
    
    # Determine chapters to test
    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(targets)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    
    logger.info(f"Found {len(chapter_dirs)} chapter directory(ies) to test")
    if chapter_dirs:
        logger.info(f"Chapters: {[chapter_slug(d, repo_root) for d in chapter_dirs]}")
    else:
        logger.warning("No chapter directories found! Check that chapter directories exist (ch1, ch2, etc.)")
    
    # Test all chapters with suite-level timeout protection
    start_time = time.time()
    all_results = []
    suite_timed_out = False
    
    def timeout_handler(signum, frame):
        nonlocal suite_timed_out
        suite_timed_out = True
        elapsed = time.time() - start_time
        logger.warning(f"SUITE TIMEOUT: Benchmark suite exceeded {suite_timeout}s timeout")
        logger.warning(f"  Elapsed time: {elapsed/3600:.2f} hours")
        logger.warning(f"  Chapters completed: {len(all_results)}")
        raise TimeoutError(f"Suite timeout after {suite_timeout} seconds")
    
    # Set up timeout signal handler
    if suite_timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(suite_timeout)
    
    try:
        for chapter_dir in chapter_dirs:
            if not chapter_dir.exists():
                continue
            
            if suite_timeout > 0:
                elapsed = time.time() - start_time
                if elapsed >= suite_timeout:
                    logger.warning("Approaching suite timeout, skipping remaining chapters")
                    break

            chapter_id = chapter_slug(chapter_dir, repo_root)
            logger.info(f"Testing chapter: {chapter_id}")
            # Call test_chapter with all flags - it handles manifest generation internally
            example_filters = chapter_filters.get(chapter_id)
            only_examples = sorted(example_filters) if example_filters else None
            result = test_chapter(
                chapter_dir=chapter_dir,
                enable_profiling=enable_profiling,
                profile_type=profile_choice if enable_profiling else "none",
                timeout_multiplier=timeout_multiplier,
                reproducible=reproducible,
                cold_start=cold_start,
                iterations=iterations,
                warmup=warmup,
                only_examples=only_examples,
                accept_regressions=accept_regressions or update_expectations,
                ncu_metric_set=ncu_metric_set,
                launch_via=launch_via,
                nproc_per_node=nproc_per_node,
                nnodes=nnodes,
                rdzv_backend=rdzv_backend,
                rdzv_endpoint=rdzv_endpoint,
                env_passthrough=torchrun_env,
                target_extra_args=parsed_extra_args,
            )
            all_results.append(result)
    except (TimeoutError, KeyboardInterrupt):
        suite_timed_out = True
        logger.warning("Benchmark suite interrupted")
    finally:
        if suite_timeout > 0:
            signal.alarm(0)
    
    # Save results
    output_json = artifact_manager.get_result_path("benchmark_test_results.json")
    output_md = artifact_manager.get_report_path("benchmark_test_results.md")
    
    if output_format in ['json', 'both']:
        with open(output_json, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'results': all_results,
            }, f, indent=2)
        logger.info(f"JSON results saved to: {output_json}")
    
    if output_format in ['markdown', 'both']:
        generate_markdown_report(all_results, output_md)
        logger.info(f"Markdown report saved to: {output_md}")
    
    # Print summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    total_benchmarks = sum(r['summary']['total_benchmarks'] for r in all_results)
    total_successful = sum(r['summary']['successful'] for r in all_results)
    total_failed = sum(r['summary']['failed'] for r in all_results)
    
    logger.info(f"Total benchmarks tested: {total_benchmarks}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_failed}")
    
    if total_benchmarks > 0:
        success_rate = (total_successful / total_benchmarks) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Calculate overall speedup statistics
    all_speedups = []
    for r in all_results:
        if r['status'] == 'completed':
            for benchmark_result in r.get('benchmarks', []):
                speedup = benchmark_result.get('speedup', 1.0)
                if speedup > 0:
                    all_speedups.append(speedup)
    
    if all_speedups:
        import statistics
        logger.info(f"\nOverall Speedup Statistics:")
        logger.info(f"  Mean: {statistics.mean(all_speedups):.2f}x")
        logger.info(f"  Median: {statistics.median(all_speedups):.2f}x")
        logger.info(f"  Min: {min(all_speedups):.2f}x")
        logger.info(f"  Max: {max(all_speedups):.2f}x")
    
    logger.info(f"\nArtifacts saved to: {artifact_manager.run_dir}")
    
    # Exit with error code if any failures
    if total_failed > 0:
        sys.exit(1)


if TYPER_AVAILABLE:
    @app.command()
    def run(
        targets: Optional[List[str]] = Option(None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to run. Repeat the flag for multiple targets. Omit or use 'all' for every chapter."),
        output_format: str = Option("both", "--format", "-f", help="Output format: 'json', 'markdown', or 'both'"),
        profile_type: str = Option("none", "--profile", "-p", help="Profiling preset: none (default), minimal, deep_dive, or roofline. Non-'none' enables nsys/ncu/PyTorch profiling.", callback=_validate_profile_type),
        suite_timeout: Optional[int] = Option(None, "--suite-timeout", help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)"),
        timeout_multiplier: float = Option(1.0, "--timeout-multiplier", help="Multiply all benchmark timeouts by this factor (e.g., 2.0 = double all timeouts)"),
        reproducible: bool = Option(False, "--reproducible", help="Enable reproducible mode: set all seeds to 42 and force deterministic algorithms (uses slower fallbacks; ops without deterministic support may error)."),
        cold_start: bool = Option(False, "--cold-start", help="Reset GPU state between benchmarks for cold start measurements"),
        iterations: Optional[int] = Option(None, "--iterations", help="Number of benchmark iterations (default: 20)"),
        warmup: Optional[int] = Option(None, "--warmup", help="Number of warmup iterations (default: 5)"),
        force_pipeline: bool = Option(False, "--force-pipeline", help="Force enable CUDA Pipeline API even on compute capability 12.0+ (may cause instability on Blackwell GPUs)"),
        artifacts_dir: Optional[str] = Option(None, "--artifacts-dir", help="Directory for artifacts (default: ./artifacts)"),
        log_level: str = Option("INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"),
        log_file: Optional[str] = Option(None, "--log-file", help="Path to log file (default: artifacts/<run_id>/logs/benchmark.log)"),
        ncu_metric_set: str = Option("auto", "--ncu-metric-set", help="Nsight Compute metric preset: auto, minimal, deep_dive, or roofline. If auto, the profile type governs metric selection.", callback=_validate_ncu_metric_set),
        accept_regressions: bool = Option(False, "--accept-regressions", help="Update expectation files when improvements are detected instead of flagging regressions.", is_flag=True),
        update_expectations: bool = Option(False, "--update-expectations", help="Force-write observed metrics into expectation files (overrides regressions). Useful for refreshing baselines on new hardware.", is_flag=True),
        launch_via: str = Option("python", "--launch-via", help="Launcher to use for benchmarks: python or torchrun."),
        nproc_per_node: Optional[int] = Option(None, "--nproc-per-node", help="torchrun --nproc_per_node value."),
        nnodes: Optional[str] = Option(None, "--nnodes", help="torchrun --nnodes value."),
        rdzv_backend: Optional[str] = Option(None, "--rdzv-backend", help="torchrun rendezvous backend (defaults to c10d when nnodes is set)."),
        rdzv_endpoint: Optional[str] = Option(None, "--rdzv-endpoint", help="torchrun rendezvous endpoint (host:port)."),
        torchrun_env: Optional[List[str]] = Option(None, "--torchrun-env", help="Environment variables to forward into torchrun launches (repeatable)."),
        target_extra_args: Optional[List[str]] = Option(None, "--target-extra-arg", help="Per-target extra args, format: target=\"--flag value\". Repeatable."),
    ):
        """Run benchmarks - discover, run, and summarize results.
        
        This is the main command for running benchmarks. It discovers baseline/optimized
        pairs, runs benchmarks with profiling, and generates reports.
        """
        _execute_benchmarks(
            targets=list(targets) if targets else None,
            output_format=output_format,
            profile_type=profile_type,
            suite_timeout=suite_timeout,
            timeout_multiplier=timeout_multiplier,
            reproducible=reproducible,
            cold_start=cold_start,
            iterations=iterations,
            warmup=warmup,
            force_pipeline=force_pipeline,
            artifacts_dir=artifacts_dir,
            log_level=log_level,
            log_file=log_file,
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            ncu_metric_set=ncu_metric_set,
            launch_via=launch_via,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            torchrun_env=torchrun_env,
            target_extra_args=target_extra_args,
        )

    @app.command()
    def verify(
        targets: Optional[List[str]] = Option(None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to verify. Repeat the flag for multiple targets. Omit or use 'all' for every chapter."),
    ):
        """Run the lightweight benchmark verification harness."""
        exit_code = run_verification(list(targets) if targets else None)
        raise typer.Exit(code=exit_code)

    @app.command("list-targets")
    def list_targets(
        chapter: Optional[str] = Option(
            None,
            "--chapter",
            "-c",
            help="Limit output to a single chapter (e.g., ch15 or labs/blackwell_matmul).",
        ),
    ):
        """List available benchmark targets in chapter:example format."""
        if chapter:
            chapter_dirs, _ = resolve_target_chapters([chapter])
        else:
            chapter_dirs = discover_all_chapters(repo_root)

        if not chapter_dirs:
            typer.echo("No chapter directories found.")
            raise typer.Exit(code=1)

        any_targets = False
        for chapter_dir in chapter_dirs:
            chapter_id = chapter_slug(chapter_dir, repo_root)
            pairs = discover_benchmarks(chapter_dir)
            if not pairs:
                continue
            any_targets = True
            for _, _, example in sorted(pairs, key=lambda entry: entry[2]):
                typer.echo(f"{chapter_id}:{example}")

        if not any_targets:
            typer.echo("No benchmark targets discovered.")


def main():
    """Entry point for CLI."""
    if not TYPER_AVAILABLE:
        print("ERROR: typer is required for CLI. Install with: pip install typer")
        sys.exit(1)
    
    if app is None:
        print("ERROR: CLI not available")
        sys.exit(1)
    
    app()


if __name__ == "__main__":
    main()
