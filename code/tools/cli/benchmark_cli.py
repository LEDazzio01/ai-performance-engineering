"""Unified CLI for benchmark execution and management using Typer."""

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

# Suppress CUDA capability warnings
warnings.filterwarnings("ignore", message=".*Found GPU.*cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

UTILITY_SCRIPTS = {
    "kv-cache": repo_root / "tools" / "utilities" / "kv_cache_calc.py",
    "cost-per-token": repo_root / "tools" / "utilities" / "calculate_cost_per_token.py",
    "compare-precision": repo_root / "tools" / "utilities" / "compare_precision_accuracy.py",
    "detect-cutlass": repo_root / "tools" / "utilities" / "detect_cutlass_info.py",
    "dump-hw": repo_root / "tools" / "utilities" / "dump_hardware_capabilities.py",
    "probe-hw": repo_root / "tools" / "utilities" / "probe_hardware_capabilities.py",
}


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
from common.python.logger import setup_logging, get_logger
from common.python.artifact_manager import ArtifactManager
from tools.verification.verify_all_benchmarks import resolve_target_chapters, run_verification
from common.python import profiler_config as profiler_config_mod
from common.python.discovery import chapter_slug, discover_all_chapters

apply_env_defaults()


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
    from common.python.chapter_compare_template import discover_benchmarks
    from tools.testing.run_all_benchmarks import test_chapter, generate_markdown_report

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Test functions come from run_all_benchmarks; if that import failed above, this is False.
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
    # Verification (on by default)
    verify_correctness: bool = True,
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

    try:
        from common.python.cuda_capabilities import set_force_pipeline

        set_force_pipeline(force_pipeline)
    except ImportError:
        pass  # cuda_capabilities not available

    artifact_manager = ArtifactManager(base_dir=Path(artifacts_dir) if artifacts_dir else None)
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

    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(targets)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"FOUND {len(chapter_dirs)} chapter(s)")

    enable_profiling = (profile_type or "none").lower() != "none"

    _apply_suite_timeout(suite_timeout)

    all_results = []
    for chapter_dir in chapter_dirs:
        chapter_id = chapter_slug(chapter_dir, repo_root)
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
            # Verification
            verify_correctness=verify_correctness,
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
        output_format: str = Option("both", "--format", "-f", help="Output format: 'json', 'markdown', or 'both'", callback=_validate_output_format),
        profile_type: str = Option("none", "--profile", "-p", help="Profiling preset: none (default), minimal, deep_dive, or roofline. Non-'none' enables nsys/ncu/PyTorch profiling.", callback=_validate_profile_type),
        suite_timeout: Optional[int] = Option(14400, "--suite-timeout", help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)"),
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
        # Verification (on by default - can skip with --skip-verify)
        skip_verify: bool = Option(False, "--skip-verify", help="Skip output correctness verification. By default, optimized outputs are verified against baseline.", is_flag=True),
    ):
        """Run benchmarks - discover, run, and summarize results."""
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
        _execute_benchmarks(
            targets=combined_targets or None,
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
            # Verification
            verify_correctness=not skip_verify,
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

    @app.command("analyze")
    def analyze(
        show_leaderboards: bool = Option(True, "--leaderboards/--no-leaderboards", help="Show separate speed/memory leaderboards"),
        show_pareto: bool = Option(True, "--pareto/--no-pareto", help="Show Pareto-optimal benchmarks"),
        show_tradeoffs: bool = Option(True, "--tradeoffs/--no-tradeoffs", help="Show cost-benefit trade-off analysis"),
        show_recommendations: bool = Option(True, "--recommendations/--no-recommendations", help="Show constraint-based recommendations"),
        show_chart: bool = Option(True, "--chart/--no-chart", help="Show ASCII trade-off scatter chart"),
        top_n: int = Option(5, "--top", "-n", help="Number of entries to show per category"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
    ):
        """Analyze benchmark results: Pareto frontier, trade-offs, and recommendations."""
        from tools.dashboard.server import DashboardHandler
        
        # Create a mock handler to access analysis methods
        class MockHandler(DashboardHandler):
            def __init__(self):
                self.data_file = None
        
        handler = MockHandler()
        
        results = {}
        
        if show_leaderboards:
            results['leaderboards'] = handler.get_categorized_leaderboards()
        if show_pareto:
            results['pareto'] = handler.get_pareto_frontier()
        if show_tradeoffs:
            results['tradeoffs'] = handler.get_tradeoff_analysis()
        if show_recommendations:
            results['recommendations'] = handler.get_constraint_recommendations()
        
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


if __name__ == "__main__":
    main()
