"""Benchmark commands mounted by aisp (`aisp bench ...`)."""

from __future__ import annotations

import json
import shlex
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

try:
    import typer
    from typer import Option

    TYPER_AVAILABLE = True
except ImportError:  # pragma: no cover - Typer is optional for docs builds
    TYPER_AVAILABLE = False
    typer = None  # type: ignore
    Option = None  # type: ignore
    Argument = None  # type: ignore
    Context = None  # type: ignore

# Ensure nvcc emits line tables for all benchmark builds
import os  # noqa: E402
os.environ["NVCCFLAGS"] = f"-lineinfo {os.environ.get('NVCCFLAGS', '')}".strip()

# Suppress CUDA capability warnings
warnings.filterwarnings("ignore", message=".*Found GPU.*cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)

# Add repo root to path (file is under core/benchmark/)
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

UTILITY_SCRIPTS = {
    "kv-cache": repo_root / "scripts" / "utilities" / "kv_cache_calc.py",
    "cost-per-token": repo_root / "scripts" / "utilities" / "calculate_cost_per_token.py",
    "compare-precision": repo_root / "scripts" / "utilities" / "compare_precision_accuracy.py",
    "detect-cutlass": repo_root / "scripts" / "utilities" / "detect_cutlass_info.py",
    "dump-hw": repo_root / "scripts" / "utilities" / "dump_hardware_capabilities.py",
    "probe-hw": repo_root / "scripts" / "utilities" / "probe_hardware_capabilities.py",
}

from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data as load_benchmark_results
from core.plugins.loader import load_plugin_apps

# Load plugins (if installed) to allow capability registration
try:
    load_plugin_apps()
except Exception:
    pass

# LLM features are always available - everything is in this single package
LLM_CAPABLE = True


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

from core.env import apply_env_defaults, dump_environment_and_capabilities
from core.utils.logger import setup_logging, get_logger
from core.benchmark.artifact_manager import ArtifactManager
from core.profiling import profiler_config as profiler_config_mod
from core.discovery import chapter_slug, discover_all_chapters, resolve_target_chapters

apply_env_defaults()


def _get_analyzer(data_file: Optional[Path] = None) -> PerformanceAnalyzer:
    """Shared analysis helper for CLI commands."""
    loader = (lambda: load_benchmark_results(data_file)) if data_file else load_benchmark_results
    return PerformanceAnalyzer(loader)


def _validate_output_format(fmt: str | None) -> str:
    if fmt is None:
        return "both"
    normalized = fmt.strip().lower()
    valid = {"json", "markdown", "both"}
    if normalized not in valid:
        message = "Output format must be one of json, markdown, or both"
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


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


def _validate_profile_type(profile: str | None) -> str:
    if profile is None:
        return "none"
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
    """Parse --target-extra-arg entries of the form target="--flag value"."""
    parsed: Dict[str, List[str]] = {}
    for entry in entries or []:
        target, sep, args = entry.partition("=")
        if not sep or not target or not args:
            continue
        parsed[target.strip()] = shlex.split(args)
    return parsed


def _apply_suite_timeout(seconds: Optional[int]) -> None:
    """Install a SIGALRM to stop the suite after a timeout."""
    if seconds is None or seconds <= 0:
        return

    def _on_timeout(signum, frame):
        raise TimeoutError(f"Benchmark suite exceeded timeout of {seconds} seconds")

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(seconds)


def _run_utility(tool: str, tool_args: Optional[List[str]]) -> int:
    """Execute a utility script with passthrough arguments."""
    script_path = UTILITY_SCRIPTS.get(tool)
    if script_path is None:
        raise ValueError(f"Unknown utility '{tool}'.")
    if not script_path.exists():
        raise FileNotFoundError(f"Utility script not found at {script_path}")

    extra_args = tool_args or []
    cmd = [sys.executable, str(script_path), *extra_args]
    result = subprocess.run(cmd)
    return result.returncode


# Import architecture optimizations early
try:
    import arch_config  # noqa: F401
except ImportError:
    pass

# Import benchmark functionality
try:
    import torch  # noqa: F401
    from core.utils.chapter_compare_template import discover_benchmarks
    from core.harness.run_benchmarks import test_chapter, generate_markdown_report

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

    # Test functions come from run_benchmarks; if that import failed above, this is False.
TEST_FUNCTIONS_AVAILABLE = BENCHMARK_AVAILABLE

# Typer app setup
if TYPER_AVAILABLE:
    app = typer.Typer(
        name="benchmark",
        help="Unified benchmark execution and management CLI",
        add_completion=False,
    )
else:
    app = None


def _execute_benchmarks(
    targets: Optional[List[str]] = None,
    bench_root: Optional[Path] = None,
    output_format: str = "both",
    profile_type: str = "none",
    suite_timeout: Optional[int] = 14400,
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
    # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
    verify_input: bool = True,
    verify_output: bool = True,
    only_cuda: bool = False,
    only_python: bool = False,
    # LLM options
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
) -> None:
    """Execute selected benchmarks with optional profiling."""
    parsed_extra_args = _parse_target_extra_args(target_extra_args)
    active_bench_root = Path(bench_root).resolve() if bench_root else repo_root

    try:
        from core.harness.cuda_capabilities import set_force_pipeline

        set_force_pipeline(force_pipeline)
    except ImportError:
        pass  # cuda_capabilities not available

    artifact_base = Path(artifacts_dir) if artifacts_dir else active_bench_root / "artifacts"
    artifact_manager = ArtifactManager(base_dir=artifact_base)
    if log_file is None:
        log_file = artifact_manager.get_log_path()

    setup_logging(level=log_level, log_file=log_file, log_format="json", use_rich=True)
    logger = get_logger(__name__)

    if not BENCHMARK_AVAILABLE or not TEST_FUNCTIONS_AVAILABLE:
        logger.error("Benchmark dependencies missing (torch/benchmark_harness or test functions).")
        sys.exit(1)

    try:
        dump_environment_and_capabilities()
    except Exception as exc:
        logger.error(f"Skipping environment/capabilities dump due to error: {exc}")

    if only_cuda and only_python:
        logger.error("Cannot combine --only-cuda and --only-python.")
        sys.exit(1)

    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(targets, bench_root=active_bench_root)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"FOUND {len(chapter_dirs)} chapter(s)")

    enable_profiling = (profile_type or "none").lower() != "none"

    _apply_suite_timeout(suite_timeout)

    all_results = []
    for chapter_dir in chapter_dirs:
        chapter_id = chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root)
        example_filters = chapter_filters.get(chapter_id)
        only_examples = sorted(example_filters) if example_filters else None
        result = test_chapter(
            chapter_dir=chapter_dir,
            enable_profiling=enable_profiling,
            profile_type=profile_type if enable_profiling else "none",
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
            only_cuda=only_cuda,
            only_python=only_python,
            # Verification - both enabled by default for valid benchmark comparisons
            verify_input=verify_input,
            verify_output=verify_output,
            # LLM options
            llm_analysis=llm_analysis or force_llm,
            force_llm=force_llm,
            llm_provider=llm_provider,
            apply_llm_patches=apply_llm_patches,
            rebenchmark_llm_patches=rebenchmark_llm_patches,
            patch_strategy=patch_strategy,
            llm_patch_retries=llm_patch_retries,
            use_llm_cache=use_llm_cache,
            llm_explain=llm_explain,
        )
        all_results.append(result)

    output_json = artifact_manager.get_result_path("benchmark_test_results.json")
    output_md = artifact_manager.get_report_path("benchmark_test_results.md")

    if output_format in ["json", "both"]:
        with open(output_json, "w") as f:
            json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": all_results}, f, indent=2)
        logger.info(f"JSON results saved to: {output_json}")
    if output_format in ["markdown", "both"]:
        generate_markdown_report(all_results, output_md)
        logger.info(f"Markdown report saved to: {output_md}")

    total_failed = sum(r.get("summary", {}).get("failed", 0) for r in all_results)
    if total_failed > 0:
        sys.exit(1)


if TYPER_AVAILABLE:

    @app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
    def run(
        ctx: typer.Context,
        targets: Optional[List[str]] = Option(None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to run. Repeat the flag for multiple targets. Omit or use 'all' for every chapter."),
        bench_root: Optional[Path] = Option(None, "--bench-root", "-r", help="Root directory to scan for benchmarks (defaults to repo root)."),
        output_format: str = Option("both", "--format", "-f", help="Output format: 'json', 'markdown', or 'both'", callback=_validate_output_format),
        profile_type: str = Option("none", "--profile", "-p", help="Profiling preset: none (default), minimal, deep_dive, or roofline. Non-'none' enables nsys/ncu/PyTorch profiling.", callback=_validate_profile_type),
        suite_timeout: Optional[int] = Option(14400, "--suite-timeout", help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)"),
        timeout_seconds: Optional[int] = Option(None, "--timeout-seconds", help="Override suite timeout in seconds (0 disables timeout)"),
        timeout_multiplier: float = Option(1.0, "--timeout-multiplier", help="Multiply all benchmark timeouts by this factor (e.g., 2.0 = double all timeouts)"),
        reproducible: bool = Option(False, "--reproducible", help="Enable reproducible mode: set all seeds to 42 and force deterministic algorithms (uses slower fallbacks; ops without deterministic support may error)."),
        cold_start: bool = Option(False, "--cold-start", help="Reset GPU state between benchmarks for cold start measurements"),
        iterations: Optional[int] = Option(None, "--iterations", help="Number of benchmark iterations (default: chapter-specific)"),
        warmup: Optional[int] = Option(None, "--warmup", help="Number of warmup iterations (default: chapter-specific)"),
        force_pipeline: bool = Option(False, "--force-pipeline", help="Force enable CUDA Pipeline API even on compute capability 12.0+ (may cause instability on Blackwell GPUs)"),
        artifacts_dir: Optional[str] = Option(None, "--artifacts-dir", help="Directory for artifacts (default: ./artifacts)"),
        log_level: str = Option("INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"),
        log_file: Optional[str] = Option(None, "--log-file", help="Path to log file (default: artifacts/<run_id>/logs/benchmark.log)"),
        ncu_metric_set: str = Option("auto", "--ncu-metric-set", help="Nsight Compute metric preset: auto, minimal, deep_dive, or roofline. If auto, the profile type governs metric selection.", callback=_validate_ncu_metric_set),
        only_cuda: bool = Option(False, "--only-cuda", help="Run only CUDA benchmarks (skip Python)."),
        only_python: bool = Option(False, "--only-python", help="Run only Python benchmarks (skip CUDA)."),
        accept_regressions: bool = Option(False, "--accept-regressions", help="Update expectation files when improvements are detected instead of flagging regressions.", is_flag=True),
        update_expectations: bool = Option(False, "--update-expectations", help="Force-write observed metrics into expectation files (overrides regressions). Useful for refreshing baselines on new hardware.", is_flag=True),
        launch_via: str = Option("python", "--launch-via", help="Launcher to use for benchmarks: python or torchrun."),
        nproc_per_node: Optional[int] = Option(None, "--nproc-per-node", help="torchrun --nproc_per_node value."),
        nnodes: Optional[str] = Option(None, "--nnodes", help="torchrun --nnodes value."),
        rdzv_backend: Optional[str] = Option(None, "--rdzv-backend", help="torchrun rendezvous backend (defaults to c10d when nnodes is set)."),
        rdzv_endpoint: Optional[str] = Option(None, "--rdzv-endpoint", help="torchrun rendezvous endpoint (host:port)."),
        torchrun_env: Optional[List[str]] = Option(None, "--torchrun-env", help="Environment variables to forward into torchrun launches (repeatable)."),
        target_extra_args: Optional[List[str]] = Option(None, "--target-extra-arg", help='Per-target extra args, format: target="--flag value". Repeatable.'),
        # LLM analysis and patching options
        llm_analysis: bool = Option(False, "--llm-analysis", help="Enable LLM-powered analysis for benchmarks with <1.1x speedup. Requires API keys in .env.local", is_flag=True),
        force_llm: bool = Option(False, "--force-llm", help="Force LLM analysis on ALL benchmarks regardless of speedup. Use to try improving even good results.", is_flag=True),
        llm_provider: Optional[str] = Option(None, "--llm-provider", help="LLM provider: 'anthropic' or 'openai'. Defaults to env LLM_PROVIDER."),
        apply_llm_patches: bool = Option(False, "--apply-llm-patches", help="Apply LLM-suggested patches to create new optimized variants. Requires --llm-analysis.", is_flag=True),
        rebenchmark_llm_patches: bool = Option(False, "--rebenchmark-llm-patches", help="Re-benchmark LLM-patched variants. Requires --apply-llm-patches.", is_flag=True),
        patch_strategy: str = Option("ast", "--patch-strategy", help="Patch strategy: 'ast' (default, AST-based) or 'fuzzy' (text matching)."),
        llm_patch_retries: int = Option(2, "--llm-patch-retries", help="Max retry attempts when LLM patch fails (syntax/runtime errors). Default: 2"),
        no_llm_cache: bool = Option(False, "--no-llm-cache", help="Disable LLM analysis caching (always re-run LLM even if cached results exist).", is_flag=True),
        llm_explain: bool = Option(False, "--llm-explain", help="Generate educational explanations for best patches (why it works, optimization techniques used). Requires --rebenchmark-llm-patches.", is_flag=True),
        # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
        skip_input_verify: bool = Option(False, "--skip-input-verify", help="Skip input equivalence verification. WARNING: Without this check, benchmark comparisons may be invalid (different workloads).", is_flag=True),
        skip_output_verify: bool = Option(False, "--skip-output-verify", help="Skip output correctness verification. WARNING: Without this check, optimizations may produce incorrect results.", is_flag=True),
        skip_verify: bool = Option(False, "--skip-verify", help="Skip BOTH input and output verification. Equivalent to --skip-input-verify --skip-output-verify.", is_flag=True),
        precheck_only: bool = Option(False, "--precheck-only", help="Validate targets and print planned command without running."),
        dry_run: bool = Option(False, "--dry-run", help="Describe planned execution without running benchmarks."),
    ):
        """Run benchmarks - discover, run, and summarize results."""
        # LLM features are always available - no capability check needed

        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root

        combined_targets: List[str] = []
        for arg in (list(targets) if targets else []):
            if arg:
                combined_targets.append(arg)
        for extra in ctx.args or []:
            if not extra:
                continue
            # Drop stray values that belong to other options when Click defers parsing.
            if extra.lower() in {"none", "minimal", "deep_dive", "roofline", "json", "markdown", "both"}:
                continue
            combined_targets.append(extra)
        # Final cleanup: drop any falsy or duplicate entries
        combined_targets = [t for t in combined_targets if t]
        # Deduplicate to avoid running the same target multiple times when
        # Click/Typer shuffles positional args.
        combined_targets = list(dict.fromkeys(combined_targets))
        effective_timeout = timeout_seconds if timeout_seconds is not None else suite_timeout
        if precheck_only or dry_run:
            plan = {
                "precheck_only": precheck_only,
                "dry_run": dry_run,
                "targets": combined_targets or ["all"],
                "bench_root": str(active_bench_root),
                "profile_type": profile_type,
                "output_format": output_format,
                "suite_timeout": effective_timeout,
            }
            typer.echo(json.dumps(plan, indent=2))
            raise typer.Exit(code=0)
        _execute_benchmarks(
            targets=combined_targets or None,
            bench_root=active_bench_root,
            output_format=output_format,
            profile_type=profile_type,
            suite_timeout=effective_timeout,
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
            only_cuda=only_cuda,
            only_python=only_python,
            launch_via=launch_via,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            torchrun_env=torchrun_env,
            target_extra_args=target_extra_args,
            # Verification - both enabled by default for valid benchmark comparisons
            verify_input=not (skip_verify or skip_input_verify),
            verify_output=not (skip_verify or skip_output_verify),
            # LLM options
            llm_analysis=llm_analysis or force_llm,
            force_llm=force_llm,
            llm_provider=llm_provider,
            apply_llm_patches=apply_llm_patches,
            rebenchmark_llm_patches=rebenchmark_llm_patches,
            patch_strategy=patch_strategy,
            llm_patch_retries=llm_patch_retries,
            use_llm_cache=not no_llm_cache,
            llm_explain=llm_explain,
        )

    @app.command("list-targets")
    def list_targets(
        chapter: Optional[str] = Option(
            None,
            "--chapter",
            "-c",
            help="Limit output to a single chapter (e.g., ch15 or labs/blackwell_matmul).",
        ),
        bench_root: Optional[Path] = Option(None, "--bench-root", "-r", help="Root directory to scan for benchmarks (defaults to repo root)."),
    ):
        """List available benchmark targets in chapter:example format."""
        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
        if chapter:
            chapter_dirs, _ = resolve_target_chapters([chapter], bench_root=active_bench_root)
        else:
            chapter_dirs = discover_all_chapters(active_bench_root, bench_roots=[active_bench_root])

        if not chapter_dirs:
            typer.echo("No chapter directories found.")
            raise typer.Exit(code=1)

        any_targets = False
        for chapter_dir in chapter_dirs:
            chapter_id = chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root)
            pairs = discover_benchmarks(chapter_dir)
            if not pairs:
                continue
            any_targets = True
            for _, _, example in sorted(pairs, key=lambda entry: entry[2]):
                typer.echo(f"{chapter_id}:{example}")

        if not any_targets:
            typer.echo("No benchmark targets discovered.")

    @app.command("list-chapters")
    def list_chapters(
        bench_root: Optional[Path] = Option(None, "--bench-root", "-r", help="Root directory to scan for benchmarks (defaults to repo root)."),
    ):
        """List all discoverable chapters and labs."""
        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
        chapter_dirs = discover_all_chapters(active_bench_root, bench_roots=[active_bench_root])
        for chapter_dir in chapter_dirs:
            typer.echo(chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root))

    @app.command("analyze")
    def analyze(
        show_leaderboards: bool = Option(True, "--leaderboards/--no-leaderboards", help="Show separate speed/memory leaderboards"),
        show_pareto: bool = Option(True, "--pareto/--no-pareto", help="Show Pareto-optimal benchmarks"),
        show_tradeoffs: bool = Option(True, "--tradeoffs/--no-tradeoffs", help="Show cost-benefit trade-off analysis"),
        show_recommendations: bool = Option(True, "--recommendations/--no-recommendations", help="Show constraint-based recommendations"),
        show_chart: bool = Option(True, "--chart/--no-chart", help="Show ASCII trade-off scatter chart"),
        top_n: int = Option(5, "--top", "-n", help="Number of entries to show per category"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """Analyze benchmark results: Pareto frontier, trade-offs, and recommendations."""
        analyzer = _get_analyzer(data_file)
        
        results = {}
        
        if show_leaderboards:
            results['leaderboards'] = analyzer.get_categorized_leaderboards()
        if show_pareto:
            results['pareto'] = analyzer.get_pareto_frontier()
        if show_tradeoffs:
            results['tradeoffs'] = analyzer.get_tradeoff_analysis()
        if show_recommendations:
            results['recommendations'] = analyzer.get_constraint_recommendations()
        
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        # Pretty print results
        typer.echo("\n" + "=" * 70)
        typer.echo("📊 MULTI-METRIC BENCHMARK ANALYSIS")
        typer.echo("=" * 70)
        
        if show_leaderboards and 'leaderboards' in results:
            boards = results['leaderboards'].get('leaderboards', {})
            
            # Speed leaderboard
            speed = boards.get('speed', {})
            typer.echo(f"\n🚀 SPEED CHAMPIONS ({speed.get('count', 0)} benchmarks)")
            typer.echo("-" * 50)
            for e in speed.get('entries', [])[:top_n]:
                rank_icon = "🥇" if e['rank'] == 1 else "🥈" if e['rank'] == 2 else "🥉" if e['rank'] == 3 else f"#{e['rank']}"
                typer.echo(f"  {rank_icon} {e['name']}: {e['primary_metric']}")
            
            # Memory leaderboard
            memory = boards.get('memory', {})
            if memory.get('entries'):
                typer.echo(f"\n💾 MEMORY CHAMPIONS ({memory.get('count', 0)} benchmarks)")
                typer.echo("-" * 50)
                for e in memory.get('entries', [])[:top_n]:
                    rank_icon = "🥇" if e['rank'] == 1 else "🥈" if e['rank'] == 2 else "🥉" if e['rank'] == 3 else f"#{e['rank']}"
                    typer.echo(f"  {rank_icon} {e['name']}: {e['primary_metric']} ({e['secondary_metric']})")
        
        if show_pareto and 'pareto' in results:
            pareto = results['pareto']
            typer.echo(f"\n⭐ PARETO-OPTIMAL BENCHMARKS ({pareto.get('pareto_count', 0)} / {pareto.get('total_count', 0)})")
            typer.echo("-" * 50)
            typer.echo("  (No other benchmark is better on ALL metrics)")
            for p in pareto.get('pareto_frontier', [])[:top_n]:
                mem_str = f"-{p['memory_savings']:.0f}% mem" if p['memory_savings'] > 0 else "N/A"
                typer.echo(f"  ⭐ {p['name']}")
                typer.echo(f"      Speed: {p['speedup']:.2f}x | Memory: {mem_str}")
        
        if show_tradeoffs and 'tradeoffs' in results:
            tradeoffs = results['tradeoffs']
            typer.echo(f"\n⚡ EFFICIENCY RANKINGS (Cost-Benefit Analysis)")
            typer.echo("-" * 50)
            
            # Memory specialists
            mem_specs = tradeoffs.get('memory_specialists', [])
            if mem_specs:
                typer.echo("  💾 Memory Efficiency:")
                for t in mem_specs[:3]:
                    typer.echo(f"      {t['name']}: {t['benefit']} ({t['cost']})")
            
            # Speed specialists
            speed_specs = tradeoffs.get('speed_specialists', [])
            if speed_specs:
                typer.echo("  🚀 Speed Efficiency (top 3):")
                for t in speed_specs[:3]:
                    typer.echo(f"      {t['name']}: {t['benefit']} (eff={t['efficiency_score']})")
        
        if show_recommendations and 'recommendations' in results:
            recs = results['recommendations']
            typer.echo(f"\n🎯 RECOMMENDATIONS BY USE CASE")
            typer.echo("-" * 50)
            for scenario in recs.get('scenarios', []):
                typer.echo(f"\n  {scenario['icon']} {scenario['name']}")
                typer.echo(f"     {scenario['description']}")
                for r in scenario.get('recommendations', [])[:2]:
                    typer.echo(f"       → {r['name']}: {r['benefit']}")
        
        if show_chart and 'pareto' in results:
            render_ascii_scatter_chart(results['pareto'])
        
        typer.echo("\n" + "=" * 70)
        typer.echo("💡 Tip: Use --json for machine-readable output")
        typer.echo("=" * 70 + "\n")
    
    def render_ascii_scatter_chart(pareto_data: dict, width: int = 60, height: int = 20):
        """Render an ASCII scatter chart of speed vs memory trade-offs."""
        all_points = pareto_data.get('all_points', [])
        pareto_points = pareto_data.get('pareto_frontier', [])
        pareto_names = set(p['name'] for p in pareto_points)
        
        if not all_points:
            return
        
        typer.echo(f"\n📈 SPEED vs MEMORY TRADE-OFF CHART")
        typer.echo("-" * 70)
        
        # Filter to reasonable range for visualization (log scale for speed)
        import math
        
        # Get ranges
        speedups = [p['speedup'] for p in all_points if p['speedup'] > 0]
        mem_savings = [p['memory_savings'] for p in all_points]
        
        if not speedups:
            typer.echo("  No data to display")
            return
        
        # Use log scale for speedup (clamped)
        min_speedup = max(0.1, min(speedups))
        max_speedup = min(1000, max(speedups))  # Cap at 1000x for visualization
        min_mem = min(mem_savings) if mem_savings else 0
        max_mem = max(mem_savings) if mem_savings else 100
        
        # Ensure some range
        if max_mem <= min_mem:
            max_mem = min_mem + 10
        
        # Create grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        for p in all_points:
            speedup = max(min_speedup, min(max_speedup, p['speedup']))
            mem = p['memory_savings']
            
            # Log scale for x (speedup)
            if speedup > 0:
                x = int((math.log10(speedup) - math.log10(min_speedup)) / 
                       (math.log10(max_speedup) - math.log10(min_speedup) + 0.001) * (width - 1))
            else:
                x = 0
            
            # Linear scale for y (memory savings)
            y = int((mem - min_mem) / (max_mem - min_mem + 0.001) * (height - 1))
            
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            y = height - 1 - y  # Flip y axis
            
            # Mark point
            if p['name'] in pareto_names:
                grid[y][x] = '★'  # Pareto optimal
            elif grid[y][x] == ' ':
                grid[y][x] = '·'  # Regular point
            elif grid[y][x] == '·':
                grid[y][x] = '○'  # Multiple points
        
        # Draw chart
        typer.echo(f"  Memory")
        typer.echo(f"  Savings")
        typer.echo(f"  {max_mem:>5.0f}% ┌" + "─" * width + "┐")
        
        for i, row in enumerate(grid):
            if i == 0 or i == height - 1 or i == height // 2:
                label = f"{max_mem - (max_mem - min_mem) * i / (height - 1):>5.0f}%"
            else:
                label = "      "
            typer.echo(f"  {label} │{''.join(row)}│")
        
        typer.echo(f"  {min_mem:>5.0f}% └" + "─" * width + "┘")
        
        # X-axis labels
        typer.echo(f"         {min_speedup:<10.1f}x" + " " * (width - 25) + f"{max_speedup:>10.1f}x")
        typer.echo(f"                              Speedup (log scale) →")
        
        # Legend
        typer.echo(f"\n  Legend: ★ = Pareto optimal  · = Regular  ○ = Multiple points")

    @app.command("summary")
    def summary(
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """Show a quick summary of benchmark results."""
        data = load_benchmark_results(data_file)
        summary = data.get("summary", {})
        benchmarks = data.get("benchmarks", [])
        payload = {
            "total": summary.get("total_benchmarks", len(benchmarks)),
            "avg_speedup": summary.get("avg_speedup", 0),
            "max_speedup": summary.get("max_speedup", 0),
            "successful": summary.get("successful"),
            "failed": summary.get("failed"),
            "memory_optimizations": summary.get("memory_optimizations"),
            "speed_optimizations": summary.get("speed_optimizations"),
            "timestamp": data.get("timestamp"),
        }
        
        if json_output:
            typer.echo(json.dumps(payload, indent=2))
            return
        
        typer.echo("\n" + "=" * 60)
        typer.echo("📜 BENCHMARK SUMMARY")
        typer.echo("=" * 60)
        typer.echo(f"Total benchmarks: {payload['total']}")
        typer.echo(f"Average speedup:  {payload['avg_speedup']:.2f}x")
        typer.echo(f"Max speedup:      {payload['max_speedup']:.2f}x")
        if payload.get("successful") is not None:
            typer.echo(f"Successful:       {payload['successful']} | Failed: {payload.get('failed', 0)}")
        if payload.get("memory_optimizations") is not None:
            typer.echo(f"Memory-focused:   {payload['memory_optimizations']} | Speed-focused: {payload.get('speed_optimizations', 0)}")
        typer.echo(f"Timestamp:        {payload.get('timestamp', 'N/A')}")

    @app.command("triage")
    def triage(
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
        baseline_file: Optional[Path] = Option(None, "--baseline", "-b", help="Optional baseline for regression detection"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        top_n: int = Option(10, "--top", "-n", help="Number of items to show"),
    ):
        """Post-benchmark triage: analyze results and get actionable recommendations.
        
        Use after running benchmarks to:
        - Identify regressions and improvements
        - Get specific tool recommendations
        - Plan next optimization steps
        """
        data = load_benchmark_results(data_file)
        benchmarks = data.get("benchmarks", [])
        
        if not benchmarks:
            typer.echo("❌ No benchmark results found. Run `aisp bench run` first.")
            raise typer.Exit(1)
        
        # Analyze
        total = len(benchmarks)
        passed = sum(1 for b in benchmarks if b.get("speedup", 0) >= 1.0)
        failed = total - passed
        avg_speedup = sum(b.get("speedup", 1.0) for b in benchmarks) / total if total > 0 else 0
        
        regressions = sorted(
            [b for b in benchmarks if b.get("speedup", 1.0) < 0.95],
            key=lambda x: x.get("speedup", 1.0)
        )[:top_n]
        
        improvements = sorted(
            [b for b in benchmarks if b.get("speedup", 1.0) > 1.05],
            key=lambda x: x.get("speedup", 1.0),
            reverse=True
        )[:top_n]
        
        slow_kernels = [b for b in benchmarks if b.get("baseline_time_ms", 0) > 100]
        
        # Build recommendations
        recommendations = []
        if regressions:
            recommendations.append({
                "tool": "aisp profile bottleneck",
                "reason": f"Identify root cause of {len(regressions)} regression(s)",
                "priority": "high"
            })
        if slow_kernels:
            recommendations.append({
                "tool": "aisp profile nsys",
                "reason": f"Profile {len(slow_kernels)} slow benchmark(s) (>100ms)",
                "priority": "medium"
            })
        if improvements:
            recommendations.append({
                "tool": "aisp bench report --output report.html",
                "reason": f"Document {len(improvements)} improvement(s)",
                "priority": "low"
            })
        recommendations.append({
            "tool": "aisp bench compare-runs",
            "reason": "Compare with previous baseline",
            "priority": "medium"
        })
        
        result = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "N/A",
                "avg_speedup": round(avg_speedup, 2),
            },
            "regressions": [
                {
                    "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": round(b.get("speedup", 0), 2),
                    "baseline_ms": round(b.get("baseline_time_ms", 0), 1),
                }
                for b in regressions
            ],
            "improvements": [
                {
                    "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": round(b.get("speedup", 0), 2),
                    "baseline_ms": round(b.get("baseline_time_ms", 0), 1),
                }
                for b in improvements
            ],
            "recommendations": recommendations,
        }
        
        if json_output:
            typer.echo(json.dumps(result, indent=2))
            return
        
        # Pretty print
        typer.echo("\n" + "=" * 60)
        typer.echo("🔍 BENCHMARK TRIAGE")
        typer.echo("=" * 60)
        
        s = result["summary"]
        typer.echo(f"\n📊 Summary: {s['total']} benchmarks | {s['pass_rate']} pass rate | {s['avg_speedup']:.2f}x avg speedup")
        
        if result["regressions"]:
            typer.echo(f"\n⚠️  REGRESSIONS ({len(result['regressions'])} found):")
            for r in result["regressions"]:
                typer.echo(f"   • {r['benchmark']}: {r['speedup']:.2f}x ({r['baseline_ms']:.1f}ms baseline)")
        else:
            typer.echo("\n✅ No regressions detected!")
        
        if result["improvements"]:
            typer.echo(f"\n🚀 TOP IMPROVEMENTS ({len(result['improvements'])} found):")
            for i in result["improvements"]:
                typer.echo(f"   • {i['benchmark']}: {i['speedup']:.2f}x ({i['baseline_ms']:.1f}ms baseline)")
        
        typer.echo(f"\n💡 RECOMMENDED NEXT STEPS:")
        for i, rec in enumerate(result["recommendations"], 1):
            priority_icon = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
            typer.echo(f"   {i}. {priority_icon} {rec['tool']}")
            typer.echo(f"      └─ {rec['reason']}")
        
        typer.echo()

    @app.command("whatif")
    def whatif(
        vram: Optional[float] = Option(None, "--vram", "-v", help="Max VRAM in GB (e.g., 24)"),
        latency: Optional[float] = Option(None, "--latency", "-l", help="Max latency in ms (e.g., 50)"),
        memory: Optional[float] = Option(None, "--memory", "-m", help="Max memory budget in GB"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """What-If Solver: Find optimizations that fit your constraints."""
        analyzer = _get_analyzer(data_file)
        params = {}
        if vram:
            params['vram'] = [str(vram)]
        if latency:
            params['latency'] = [str(latency)]
        if memory:
            params['memory_budget'] = [str(memory)]
        
        results = analyzer.get_whatif_recommendations(params)
        
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        typer.echo("\n" + "=" * 60)
        typer.echo("🔍 WHAT-IF CONSTRAINT SOLVER")
        typer.echo("=" * 60)
        
        constraints = results.get('constraints', {})
        active_constraints = [f"{k}={v}" for k, v in constraints.items() if v]
        if active_constraints:
            typer.echo(f"\nConstraints: {', '.join(active_constraints)}")
        else:
            typer.echo("\nNo constraints specified - showing all optimizations")
        
        typer.echo(f"Matching: {results.get('matching_count', 0)} / {results.get('total_benchmarks', 0)} benchmarks")
        
        if results.get('best_for_speed'):
            typer.echo(f"\n🚀 Best for Speed: {results['best_for_speed']['name']} ({results['best_for_speed']['speedup']:.2f}x)")
        if results.get('best_for_memory'):
            mem = results['best_for_memory']
            if mem['memory_savings_pct']:
                typer.echo(f"💾 Best for Memory: {mem['name']} (-{mem['memory_savings_pct']:.0f}%)")
        
        typer.echo(f"\n📋 Top Recommendations:")
        typer.echo("-" * 60)
        for i, r in enumerate(results.get('recommendations', [])[:8], 1):
            mem_str = f"-{r['memory_savings_pct']:.0f}% mem" if r['memory_savings_pct'] else ""
            typer.echo(f"  {i}. {r['name']}: {r['speedup']:.2f}x {mem_str}")
    
    @app.command("stacking")
    def stacking(
        top_n: int = Option(5, "--top", "-n", help="Number of items to show per section"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """Show which optimizations combine well together."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_optimization_stacking()

        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        typer.echo("\n" + "=" * 60)
        typer.echo("🔗 OPTIMIZATION STACKING GUIDE")
        typer.echo("=" * 60)
        
        typer.echo("\n✅ COMPATIBLE COMBINATIONS:")
        typer.echo("-" * 60)
        for c in results.get('compatible_combinations', [])[:top_n]:
            typer.echo(f"  {c['opt1']} + {c['opt2']}")
            typer.echo(f"    Synergy: {c['synergy']} | {c['benefit']}")

        typer.echo("\n❌ INCOMPATIBLE COMBINATIONS:")
        typer.echo("-" * 60)
        for c in results.get('incompatible_combinations', [])[:top_n]:
            typer.echo(f"  {c['opt1']} + {c['opt2']}")
            typer.echo(f"    Reason: {c['reason']}")

        typer.echo("\n🎯 RECOMMENDED STACKS:")
        typer.echo("-" * 60)
        for s in results.get('recommended_stacks', [])[:top_n]:
            typer.echo(f"  {s['name']}:")
            typer.echo(f"    Stack: {' → '.join(s['stack'])}")
            typer.echo(f"    Expected: {s['expected_benefit']}")
    
    @app.command("power")
    def power(
        top_n: int = Option(10, "--top", "-n", help="Number of entries to show"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """Analyze power efficiency (ops/watt) of benchmarks."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_power_efficiency()
        
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        typer.echo("\n" + "=" * 60)
        typer.echo("⚡ POWER EFFICIENCY ANALYSIS")
        typer.echo("=" * 60)
        
        typer.echo(f"\nBenchmarks with power data: {results.get('total_benchmarks_with_power', 0)}")
        typer.echo(f"Average power: {results.get('avg_power_w', 0):.1f}W")
        
        if results.get('most_efficient'):
            e = results['most_efficient']
            typer.echo(f"\n🏆 Most Efficient: {e['name']}")
            typer.echo(f"   {e['ops_per_watt']:.4f} ops/watt | {e['power_w']:.0f}W | {e['speedup']:.2f}x")
        
        typer.echo(f"\n📊 Efficiency Rankings (ops/watt):")
        typer.echo("-" * 60)
        for i, e in enumerate(results.get('efficiency_rankings', [])[:top_n], 1):
            typer.echo(f"  {i:2}. {e['name'][:40]:<40} {e['ops_per_watt']:.4f}")
    
    @app.command("cost")
    def cost(
        top_n: int = Option(10, "--top", "-n", help="Number of entries to show"),
        gpu: str = Option("B200", "--gpu", "-g", help="GPU type (B200, H100, A100, L40S, A10G, T4)"),
        rate: Optional[float] = Option(None, "--rate", "-r", help="Custom hourly rate in $/hr"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """Calculate cost savings ($/token) for optimizations."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_cost_analysis(gpu=gpu, custom_rate=rate)
        
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        typer.echo("\n" + "=" * 60)
        typer.echo("💰 COST ANALYSIS")
        typer.echo("=" * 60)
        
        typer.echo(f"\nAssuming: {results.get('assumed_gpu', 'B200')} @ ${results.get('hourly_rate', 0):.2f}/hr")
        
        if results.get('highest_savings'):
            h = results['highest_savings']
            typer.echo(f"\n🏆 Highest Savings: {h['name']}")
            typer.echo(f"   ${h['baseline_cost_per_m']:.4f} → ${h['optimized_cost_per_m']:.4f} per 1M ops")
            typer.echo(f"   Savings: ${h['savings_per_m']:.4f}/1M ops ({h['savings_pct']:.0f}%)")
        
        typer.echo(f"\n📊 Cost Savings Rankings:")
        typer.echo("-" * 60)
        typer.echo(f"  {'Benchmark':<40} {'Savings':>10}")
        typer.echo("-" * 60)
        for c in results.get('cost_rankings', [])[:top_n]:
            typer.echo(f"  {c['name'][:40]:<40} {c['savings_pct']:>8.0f}%")
    
    @app.command("scaling")
    def scaling(
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """Analyze how optimizations scale with workload size."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_scaling_analysis()
        
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        
        typer.echo("\n" + "=" * 60)
        typer.echo("📈 SCALING ANALYSIS")
        typer.echo("=" * 60)
        
        typer.echo(f"\n💡 Key Insight: {results.get('key_insight', '')}")
        
        typer.echo("\n📊 Top Optimizations by Category:")
        typer.echo("-" * 60)
        for cat, benchmarks in results.get('categories', {}).items():
            if benchmarks:
                typer.echo(f"\n  {cat.upper()}:")
                for b in benchmarks[:3]:
                    typer.echo(f"    • {b['name']}: {b['speedup']:.2f}x")
        
        typer.echo("\n🎯 Scaling Recommendations:")
        typer.echo("-" * 60)
        for r in results.get('scaling_recommendations', []):
            typer.echo(f"\n  {r['factor']}:")
            typer.echo(f"    {r['insight']}")
            typer.echo(f"    → {r['recommendation']}")

    @app.command("report")
    def report(
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
        output: Path = Option(Path("report.pdf"), "--output", "-o", help="Output file (.pdf or .html)"),
        format: str = Option("pdf", "--format", "-f", help="pdf or html"),
        title: str = Option("GPU Performance Report", "--title", help="Report title"),
        author: str = Option("AI Performance Engineering", "--author", help="Author/owner"),
    ):
        """Generate PDF/HTML report from benchmark results."""
        try:
            from core.analysis.reporting.generator import generate_report, ReportConfig
        except Exception as exc:  # pragma: no cover - optional dependency
            typer.echo(f"Report generation unavailable: {exc}", err=True)
            raise typer.Exit(code=1)

        input_path = str(data_file) if data_file else "benchmark_test_results.json"
        cfg = ReportConfig(title=title, author=author)
        path = generate_report(input_path, str(output), format=format, config=cfg)
        typer.echo(f"✅ Report generated: {path}")

    @app.command("export")
    def export(
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
        format: str = Option("csv", "--format", "-f", help="csv, markdown, or json"),
        output: Optional[Path] = Option(None, "--output", "-o", help="Output file path"),
    ):
        """Export benchmark results to csv/markdown/json."""
        analyzer = _get_analyzer(data_file)
        data = analyzer._load_data()  # type: ignore[attr-defined]  # lightweight export
        fmt = format.lower()
        out_path = output or Path(f"export.{fmt}")
        benchmarks = data.get("benchmarks", [])

        if fmt == "json":
            out_path.write_text(json.dumps(data, indent=2))
        elif fmt == "markdown":
            lines = ["| Benchmark | Speedup | Baseline (ms) | Type |", "|---|---|---|---|"]
            for b in benchmarks:
                lines.append(
                    f"| {b.get('chapter')}:{b.get('name')} | {b.get('speedup', 0):.2f}x | {b.get('baseline_time_ms', 0):.3f} | {b.get('type', 'python')} |"
                )
            out_path.write_text("\n".join(lines))
        elif fmt == "csv":
            import csv

            with out_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["benchmark", "speedup", "baseline_ms", "type"])
                for b in benchmarks:
                    writer.writerow(
                        [
                            f"{b.get('chapter')}:{b.get('name')}",
                            f"{b.get('speedup', 0):.2f}",
                            f"{b.get('baseline_time_ms', 0):.3f}",
                            b.get("type", "python"),
                        ]
                    )
        else:
            typer.echo("Format must be one of: csv, markdown, json", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"✅ Exported to {out_path}")

    @app.command("compare-runs")
    def compare_runs(
        baseline: Path = Option(..., "--baseline", "-b", help="Baseline benchmark_test_results.json"),
        candidate: Path = Option(..., "--candidate", "-c", help="Candidate benchmark_test_results.json"),
        top: int = Option(10, "--top", "-n", help="Top regressions/improvements to show"),
    ):
        """Diff two benchmark JSON files and show speedup deltas."""
        def _load(path: Path) -> dict:
            with open(path) as f:
                return json.load(f)

        base = _load(baseline)
        cand = _load(candidate)

        def _flatten(blob: dict) -> dict:
            flat = {}
            for chapter in blob.get("results", []):
                chap = chapter.get("chapter", "unknown")
                for bench in chapter.get("benchmarks", []):
                    key = f"{chap}:{bench.get('example', bench.get('name', 'unknown'))}"
                    flat[key] = bench
            return flat

        bflat = _flatten(base)
        cflat = _flatten(cand)
        deltas = []
        for key, c in cflat.items():
            b = bflat.get(key)
            if not b:
                continue
            delta = (c.get("best_speedup", 0) or 0) - (b.get("best_speedup", 0) or 0)
            deltas.append((key, delta, b.get("best_speedup", 0) or 0, c.get("best_speedup", 0) or 0))

        deltas.sort(key=lambda x: x[1])
        regressions = [d for d in deltas if d[1] < 0][:top]
        improvements = sorted([d for d in deltas if d[1] > 0], key=lambda x: -x[1])[:top]

        typer.echo("\n🚨 Regressions")
        for name, delta, b, c in regressions:
            typer.echo(f"  {name}: {b:.2f}x → {c:.2f}x (Δ {delta:.2f})")

        typer.echo("\n✅ Improvements")
        for name, delta, b, c in improvements:
            typer.echo(f"  {name}: {b:.2f}x → {c:.2f}x (Δ +{delta:.2f})")

        typer.echo("\nℹ️ Use --top to adjust output; full data available in JSON files.")
    
    @app.command("tui")
    def tui(
        simple: bool = Option(False, "--simple", "-s", help="Use simple menu instead of curses TUI"),
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ):
        """Interactive Terminal UI for benchmark analysis."""
        if simple:
            _run_basic_menu(data_file)
            return
        
        try:
            from cli.tui import run_tui
            run_tui(data_file)
        except KeyboardInterrupt:
            typer.echo("\nExiting...")
        except ImportError as e:
            typer.echo(f"TUI requires curses (standard on Unix): {e}")
            typer.echo("Falling back to basic menu...")
            _run_basic_menu(data_file)
        except Exception as e:
            typer.echo(f"TUI error: {e}")
            typer.echo("Falling back to basic menu...")
            _run_basic_menu(data_file)
    
    def _run_basic_menu(data_file: Optional[Path] = None):
        """Simple menu-based interface when rich TUI isn't available."""
        analyzer = _get_analyzer(data_file)
        
        while True:
            typer.echo("\n" + "=" * 50)
            typer.echo("📊 BENCHMARK ANALYSIS MENU")
            typer.echo("=" * 50)
            typer.echo("  1. Leaderboards (speed vs memory)")
            typer.echo("  2. Pareto Frontier")
            typer.echo("  3. What-If Solver")
            typer.echo("  4. Optimization Stacking")
            typer.echo("  5. Power Efficiency")
            typer.echo("  6. Cost Analysis")
            typer.echo("  7. Scaling Analysis")
            typer.echo("  8. Trade-off Chart (ASCII)")
            typer.echo("  q. Quit")
            typer.echo("-" * 50)
            
            choice = input("Select option: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == '1':
                data = analyzer.get_categorized_leaderboards()
                _print_leaderboards(data)
            elif choice == '2':
                data = analyzer.get_pareto_frontier()
                _print_pareto(data)
            elif choice == '3':
                vram = input("Max VRAM (GB, or Enter to skip): ").strip()
                latency = input("Max latency (ms, or Enter to skip): ").strip()
                params = {}
                if vram:
                    params['vram'] = [vram]
                if latency:
                    params['latency'] = [latency]
                data = analyzer.get_whatif_recommendations(params)
                _print_whatif(data)
            elif choice == '4':
                data = analyzer.get_optimization_stacking()
                _print_stacking(data)
            elif choice == '5':
                data = analyzer.get_power_efficiency()
                _print_power(data)
            elif choice == '6':
                data = analyzer.get_cost_analysis()
                _print_cost(data)
            elif choice == '7':
                data = analyzer.get_scaling_analysis()
                _print_scaling(data)
            elif choice == '8':
                data = analyzer.get_pareto_frontier()
                render_ascii_scatter_chart(data)
    
    def _run_interactive_tui():
        """Rich interactive TUI using curses or similar."""
        # For now, fall back to basic menu
        _run_basic_menu()
    
    def _print_leaderboards(data):
        boards = data.get('leaderboards', {})
        for name, board in boards.items():
            typer.echo(f"\n{board.get('title', name)}:")
            for e in board.get('entries', [])[:5]:
                typer.echo(f"  #{e['rank']} {e['name']}: {e['primary_metric']}")
    
    def _print_pareto(data):
        typer.echo(f"\n⭐ Pareto Optimal: {data.get('pareto_count', 0)} / {data.get('total_count', 0)}")
        for p in data.get('pareto_frontier', [])[:5]:
            typer.echo(f"  ⭐ {p['name']}: {p['speedup']:.2f}x, {p['memory_savings']:.0f}% mem")
    
    def _print_whatif(data):
        typer.echo(f"\nMatching: {data.get('matching_count', 0)} benchmarks")
        for r in data.get('recommendations', [])[:5]:
            typer.echo(f"  • {r['name']}: {r['speedup']:.2f}x")
    
    def _print_stacking(data):
        typer.echo("\n✅ Compatible:")
        for c in data.get('compatible_combinations', [])[:3]:
            typer.echo(f"  {c['opt1']} + {c['opt2']}: {c['benefit']}")
    
    def _print_power(data):
        typer.echo(f"\nBenchmarks with power data: {data.get('total_benchmarks_with_power', 0)}")
        for e in data.get('efficiency_rankings', [])[:5]:
            typer.echo(f"  {e['name']}: {e['ops_per_watt']:.4f} ops/W")
    
    def _print_cost(data):
        typer.echo(f"\nAssuming ${data.get('hourly_rate', 0):.2f}/hr")
        for c in data.get('cost_rankings', [])[:5]:
            typer.echo(f"  {c['name']}: {c['savings_pct']:.0f}% savings")
    
    def _print_scaling(data):
        for r in data.get('scaling_recommendations', [])[:3]:
            typer.echo(f"\n{r['factor']}: {r['recommendation']}")

    @app.command("utils")
    def utils(
        tool: str = Option(
            ...,
            "--tool",
            "-u",
            help=f"Utility to run. Available: {', '.join(sorted(UTILITY_SCRIPTS))}",
        ),
        tool_args: Optional[List[str]] = typer.Argument(
            None,
            help="Arguments forwarded to the utility (use -- to separate).",
        ),
    ):
        """Run repository utilities (e.g., KV cache calculator) from one entrypoint."""
        try:
            exit_code = _run_utility(tool, tool_args)
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)
        except FileNotFoundError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)

        raise typer.Exit(code=exit_code)


def main():
    """Entry point for CLI."""
    if not TYPER_AVAILABLE:
        print("ERROR: typer is required for CLI. Install with: pip install typer")
        sys.exit(1)

    if app is None:
        print("ERROR: CLI not available")
        sys.exit(1)

    app()
