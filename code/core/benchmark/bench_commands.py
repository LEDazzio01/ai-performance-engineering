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
    pm_sampling_interval: Optional[int] = None,
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
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            ncu_metric_set=ncu_metric_set,
            pm_sampling_interval=pm_sampling_interval,
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
        pm_sampling_interval: Optional[int] = Option(
            None,
            "--pm-sampling-interval",
            help="Nsight Compute pm-sampling-interval (cycles between samples). Optional: set to reduce overhead; defaults to unset.",
        ),
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
        verify_phase: str = Option("gate", "--verify-phase", help="Verification enforcement phase: 'detect' (report only), 'quarantine' (exclude non-compliant from reports), 'gate' (default, fail on verification failure)"),
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
        # Set verification enforcement phase
        if verify_phase:
            os.environ["VERIFY_ENFORCEMENT_PHASE"] = verify_phase.lower()
        
        if precheck_only or dry_run:
            plan = {
                "precheck_only": precheck_only,
                "dry_run": dry_run,
                "targets": combined_targets or ["all"],
                "bench_root": str(active_bench_root),
                "profile_type": profile_type,
                "output_format": output_format,
                "suite_timeout": effective_timeout,
                "verify_phase": verify_phase,
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

    @app.command("verify")
    def verify(
        targets: Optional[List[str]] = Option(None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to verify. Repeat the flag for multiple targets. Omit or use 'all' for every chapter."),
        bench_root: Optional[Path] = Option(None, "--bench-root", "-r", help="Root directory to scan for benchmarks (defaults to repo root)."),
        verify_phase: str = Option("gate", "--verify-phase", "-p", help="Verification enforcement phase: 'detect' (report only), 'quarantine' (exclude non-compliant), 'gate' (default, strict enforcement)"),
        skip_jitter: bool = Option(False, "--skip-jitter", help="Skip jitter check (output changes when inputs are perturbed)", is_flag=True),
        skip_fresh_input: bool = Option(False, "--skip-fresh-input", help="Skip fresh-input check (different seeds produce different outputs)", is_flag=True),
        skip_workload: bool = Option(False, "--skip-workload", help="Skip workload invariant check (bytes/tokens/ops per iteration)", is_flag=True),
        json_output: bool = Option(False, "--json", help="Output results as JSON", is_flag=True),
        verbose: bool = Option(False, "--verbose", "-v", help="Verbose output with detailed comparison info", is_flag=True),
        clear_cache: bool = Option(False, "--clear-cache", help="Clear golden output cache before verification", is_flag=True),
    ):
        """Verify benchmark pairs for correctness without measuring performance.
        
        Runs verification checks on baseline/optimized benchmark pairs:
        - Input signature matching (same batch size, dtypes, shapes)
        - Output correctness (optimized produces same outputs as baseline)
        - Jitter check (outputs change when inputs are perturbed)
        - Fresh-input check (different seeds produce different outputs)
        - Workload invariant check (same bytes/tokens/ops per iteration)
        
        Examples:
            aisp bench verify                     # Verify all chapters
            aisp bench verify -t ch11            # Verify chapter 11
            aisp bench verify -t ch11:streams    # Verify specific example
            aisp bench verify --json             # JSON output for CI
        """
        from core.benchmark.verify_runner import VerifyRunner, VerifyConfig
        from core.benchmark.verification import EnforcementPhase, QuarantineReason, get_enforcement_phase
        from core.benchmark.quarantine import QuarantineManager
        from core.discovery import discover_benchmarks
        import importlib.util
        
        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
        
        # Set enforcement phase
        os.environ["VERIFY_ENFORCEMENT_PHASE"] = verify_phase.lower()
        phase = get_enforcement_phase()
        
        # Initialize verification components
        cache_dir = active_bench_root / "artifacts" / "verify_cache"
        quarantine_mgr = QuarantineManager(cache_dir=cache_dir)
        
        if clear_cache:
            import shutil
            golden_cache = cache_dir / "golden_outputs"
            if golden_cache.exists():
                shutil.rmtree(golden_cache)
                typer.echo(f"🗑️  Cleared golden output cache: {golden_cache}")
        
        verify_config = VerifyConfig(
            skip_jitter_check=skip_jitter,
            skip_fresh_input_check=skip_fresh_input,
            seed=42,
        )
        verify_runner = VerifyRunner(
            cache_dir=cache_dir / "golden_outputs",
            quarantine_manager=quarantine_mgr,
        )
        
        # Resolve targets
        effective_targets = targets if targets else ["all"]
        chapters_dict = discover_all_chapters(active_bench_root)
        resolved = resolve_target_chapters(effective_targets, chapters_dict)
        
        results = []
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for chapter_slug_name, chapter_data in resolved.items():
            chapter_dir = chapter_data.get("chapter_dir")
            if not chapter_dir:
                continue
            
            typer.echo(f"\n{'='*60}")
            typer.echo(f"📋 Verifying {chapter_slug_name}")
            typer.echo(f"{'='*60}")
            
            try:
                pairs = discover_benchmarks(Path(chapter_dir))
            except Exception as e:
                typer.echo(f"  ⚠️  Could not find benchmark pairs: {e}")
                skipped_count += 1
                results.append({
                    "chapter": chapter_slug_name,
                    "status": "skipped",
                    "reason": str(e),
                })
                continue
            
            if not pairs:
                typer.echo(f"  ⏭️  No benchmark pairs found")
                skipped_count += 1
                results.append({
                    "chapter": chapter_slug_name,
                    "status": "skipped",
                    "reason": "No benchmark pairs found",
                })
                continue
            
            def _load_benchmark_module(path: Path):
                """Load a Python benchmark module by path."""
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load spec for {path}")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

            def _format_reason(reason: object) -> str:
                if reason is None:
                    return "unknown"
                value = getattr(reason, "value", None)
                if value is not None:
                    return str(value)
                return str(reason)

            def _to_quarantine_reason(reason: object) -> Optional[QuarantineReason]:
                if isinstance(reason, QuarantineReason):
                    return reason
                value = getattr(reason, "value", None)
                if isinstance(value, str):
                    try:
                        return QuarantineReason(value)
                    except Exception:
                        return None
                if isinstance(reason, str):
                    try:
                        return QuarantineReason(reason)
                    except Exception:
                        return None
                return None
            
            for baseline_path, optimized_paths, example_name in pairs:
                # Verify baseline once, then verify all optimized variants against cached golden output.
                try:
                    baseline_mod = _load_benchmark_module(baseline_path)
                except Exception as e:
                    for optimized_path in optimized_paths:
                        pair_name = f"{example_name}/{optimized_path.stem.replace('optimized_', '')}"
                        typer.echo(f"\n  🔍 {pair_name}:")
                        typer.echo(f"      Baseline:  {baseline_path.name}")
                        typer.echo(f"      Optimized: {optimized_path.name}")
                        typer.echo(f"      ❌ ERROR: Failed to load baseline module: {e}")
                        failed_count += 1
                        results.append({
                            "chapter": chapter_slug_name,
                            "pair": pair_name,
                            "status": "error",
                            "reason": f"Failed to load baseline module: {e}",
                        })
                    continue

                baseline_cls = getattr(baseline_mod, "Benchmark", None)
                if not baseline_cls:
                    for optimized_path in optimized_paths:
                        pair_name = f"{example_name}/{optimized_path.stem.replace('optimized_', '')}"
                        typer.echo(f"\n  🔍 {pair_name}:")
                        typer.echo(f"      Baseline:  {baseline_path.name}")
                        typer.echo(f"      Optimized: {optimized_path.name}")
                        typer.echo("      ❌ Missing Benchmark class (baseline)")
                        failed_count += 1
                        results.append({
                            "chapter": chapter_slug_name,
                            "pair": pair_name,
                            "status": "failed",
                            "reason": "Missing Benchmark class (baseline)",
                        })
                    continue

                try:
                    baseline = baseline_cls()
                except Exception as e:
                    for optimized_path in optimized_paths:
                        pair_name = f"{example_name}/{optimized_path.stem.replace('optimized_', '')}"
                        typer.echo(f"\n  🔍 {pair_name}:")
                        typer.echo(f"      Baseline:  {baseline_path.name}")
                        typer.echo(f"      Optimized: {optimized_path.name}")
                        typer.echo(f"      ❌ ERROR: Failed to instantiate baseline: {e}")
                        failed_count += 1
                        results.append({
                            "chapter": chapter_slug_name,
                            "pair": pair_name,
                            "status": "error",
                            "reason": f"Failed to instantiate baseline: {e}",
                        })
                    continue

                baseline_result = verify_runner.verify_baseline(baseline, config=verify_config)
                baseline_reason = _format_reason(baseline_result.reason)
                baseline_skipped = (not baseline_result.passed) and baseline_reason.startswith("SKIPPED:")

                if not baseline_result.passed:
                    if phase in (EnforcementPhase.QUARANTINE, EnforcementPhase.GATE):
                        qr = _to_quarantine_reason(baseline_result.reason)
                        if qr is not None:
                            quarantine_mgr.quarantine(str(baseline_path), qr)

                for optimized_path in optimized_paths:
                    pair_name = f"{example_name}/{optimized_path.stem.replace('optimized_', '')}"
                    typer.echo(f"\n  🔍 {pair_name}:")
                    typer.echo(f"      Baseline:  {baseline_path.name}")
                    typer.echo(f"      Optimized: {optimized_path.name}")
                    
                    try:
                        if not baseline_result.passed:
                            status = "skipped" if baseline_skipped else "failed"
                            if baseline_skipped:
                                typer.echo(f"      ⏭️  SKIPPED: {baseline_reason}")
                                skipped_count += 1
                            else:
                                typer.echo(f"      ❌ FAILED: {baseline_reason}")
                                failed_count += 1
                            results.append({
                                "chapter": chapter_slug_name,
                                "pair": pair_name,
                                "status": status,
                                "reason": baseline_reason,
                            })
                            continue

                        # Load optimized benchmark
                        optimized_mod = _load_benchmark_module(optimized_path)
                        
                        optimized_cls = getattr(optimized_mod, "Benchmark", None)
                        
                        if not optimized_cls:
                            typer.echo(f"      ❌ Missing Benchmark class")
                            failed_count += 1
                            results.append({
                                "chapter": chapter_slug_name,
                                "pair": pair_name,
                                "status": "failed",
                                "reason": "Missing Benchmark class",
                            })
                            continue
                        
                        optimized = optimized_cls()

                        # Validate timing config matches (anti-gaming)
                        if not verify_config.skip_timing_validation:
                            timing_valid, timing_error = verify_runner._validate_timing_config(baseline, optimized)  # noqa: SLF001
                            if not timing_valid:
                                typer.echo(f"      ❌ FAILED: {QuarantineReason.TIMING_CONFIG_MISMATCH.value}")
                                if verbose and timing_error:
                                    typer.echo(f"         Details: {timing_error}")
                                failed_count += 1
                                results.append({
                                    "chapter": chapter_slug_name,
                                    "pair": pair_name,
                                    "status": "failed",
                                    "reason": QuarantineReason.TIMING_CONFIG_MISMATCH.value,
                                    "details": timing_error,
                                })
                                if phase in (EnforcementPhase.QUARANTINE, EnforcementPhase.GATE):
                                    quarantine_mgr.quarantine(str(optimized_path), QuarantineReason.TIMING_CONFIG_MISMATCH)
                                continue

                        # Run verification (baseline golden output already cached)
                        result = verify_runner.verify_optimized(optimized, config=verify_config)
                        
                        if result.passed:
                            typer.echo(f"      ✅ PASSED")
                            passed_count += 1
                            results.append({
                                "chapter": chapter_slug_name,
                                "pair": pair_name,
                                "status": "passed",
                            })
                        else:
                            reason_str = _format_reason(result.reason)
                            is_skipped = reason_str.startswith("SKIPPED:")
                            if is_skipped:
                                typer.echo(f"      ⏭️  SKIPPED: {reason_str}")
                                skipped_count += 1
                            else:
                                typer.echo(f"      ❌ FAILED: {reason_str}")
                                failed_count += 1
                            if verbose and result.details:
                                typer.echo(f"         Details: {result.details}")
                            results.append({
                                "chapter": chapter_slug_name,
                                "pair": pair_name,
                                "status": "skipped" if is_skipped else "failed",
                                "reason": reason_str,
                                "details": result.details,
                            })
                            
                            # Quarantine if in quarantine or gate phase
                            if (not is_skipped) and phase in (EnforcementPhase.QUARANTINE, EnforcementPhase.GATE):
                                qr = _to_quarantine_reason(result.reason)
                                if qr is not None:
                                    quarantine_mgr.quarantine(str(optimized_path), qr)
                    
                    except Exception as e:
                        typer.echo(f"      ❌ ERROR: {e}")
                        if verbose:
                            import traceback
                            typer.echo(f"         {traceback.format_exc()}")
                        failed_count += 1
                        results.append({
                            "chapter": chapter_slug_name,
                            "pair": pair_name,
                            "status": "error",
                            "reason": str(e),
                        })
        
        # Summary
        typer.echo(f"\n{'='*60}")
        typer.echo(f"📊 VERIFICATION SUMMARY")
        typer.echo(f"{'='*60}")
        typer.echo(f"  ✅ Passed:  {passed_count}")
        typer.echo(f"  ❌ Failed:  {failed_count}")
        typer.echo(f"  ⏭️  Skipped: {skipped_count}")
        typer.echo(f"  📈 Total:   {passed_count + failed_count + skipped_count}")
        
        if failed_count > 0:
            typer.echo(f"\n  ⚠️  Phase: {phase.value}")
            if phase == EnforcementPhase.GATE:
                typer.echo(f"  🚫 Gate mode: non-compliant benchmarks will block performance measurement")
            elif phase == EnforcementPhase.QUARANTINE:
                typer.echo(f"  🏷️  Quarantine mode: non-compliant benchmarks excluded from reports")
            else:
                typer.echo(f"  ℹ️  Detect mode: issues reported but not enforced")
        
        if json_output:
            output = {
                "summary": {
                    "passed": passed_count,
                    "failed": failed_count,
                    "skipped": skipped_count,
                    "phase": phase.value,
                },
                "results": results,
                "quarantine": {k: v.to_dict() for k, v in quarantine_mgr.get_all_records().items()} if hasattr(quarantine_mgr, 'get_all_records') else {},
            }
            typer.echo("\n" + json.dumps(output, indent=2, default=str))
        
        # Exit with error if failed and in gate mode
        if failed_count > 0 and phase == EnforcementPhase.GATE:
            raise typer.Exit(code=1)

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

    @app.command("verify-report")
    def verify_report(
        data_file: Optional[Path] = Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
        output: Path = Option(Path("verification_report.html"), "--output", "-o", help="Output file (.html or .json)"),
        format: str = Option("html", "--format", "-f", help="html or json"),
        gpu: str = Option("", "--gpu", "-g", help="GPU name for theoretical peak (e.g., H100, B200)"),
        title: str = Option("Benchmark Verification Report", "--title", help="Report title"),
        quarantine: Optional[Path] = Option(None, "--quarantine", "-q", help="Path to quarantine.json"),
    ):
        """Generate verification report with anti-cheat status and theoretical peaks."""
        try:
            from core.analysis.reporting.verification_report import generate_verification_report
        except ImportError as exc:
            typer.echo(f"Verification report generation unavailable: {exc}", err=True)
            raise typer.Exit(code=1)
        
        input_path = str(data_file) if data_file else "benchmark_test_results.json"
        fmt = format.lower()
        if fmt not in ("html", "json"):
            typer.echo("Format must be html or json", err=True)
            raise typer.Exit(code=1)
        
        try:
            path = generate_verification_report(
                input_path,
                str(output),
                format=fmt,
                gpu_name=gpu,
                title=title,
                quarantine_path=quarantine,
            )
            typer.echo(f"✅ Verification report generated: {path}")
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    @app.command("theoretical-peak")
    def theoretical_peak(
        gpu: str = Option(..., "--gpu", "-g", help="GPU name (e.g., H100, B200, A100)"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
    ):
        """Show theoretical peak performance for a GPU."""
        try:
            from core.analysis.reporting.verification_report import get_theoretical_peak, GPU_THEORETICAL_PEAKS
        except ImportError as exc:
            typer.echo(f"Theoretical peak lookup unavailable: {exc}", err=True)
            raise typer.Exit(code=1)
        
        if gpu.lower() == "list":
            typer.echo("Available GPUs:")
            for key in sorted(GPU_THEORETICAL_PEAKS.keys()):
                p = GPU_THEORETICAL_PEAKS[key]
                typer.echo(f"  {key}: {p.gpu_name} ({p.architecture})")
            return
        
        peak = get_theoretical_peak(gpu)
        if not peak:
            typer.echo(f"GPU '{gpu}' not found. Use --gpu list to see available GPUs.", err=True)
            raise typer.Exit(code=1)
        
        if json_output:
            import json as json_module
            typer.echo(json_module.dumps(peak.to_dict(), indent=2))
        else:
            typer.echo(f"\n🎯 Theoretical Peak: {peak.gpu_name}")
            typer.echo(f"   Architecture: {peak.architecture}")
            typer.echo(f"   SMs: {peak.sm_count}")
            typer.echo("")
            typer.echo("   Compute:")
            typer.echo(f"     FP32:  {peak.fp32_tflops:,.1f} TFLOPS")
            typer.echo(f"     FP16:  {peak.fp16_tflops:,.1f} TFLOPS")
            typer.echo(f"     BF16:  {peak.bf16_tflops:,.1f} TFLOPS")
            if peak.fp8_tflops > 0:
                typer.echo(f"     FP8:   {peak.fp8_tflops:,.1f} TFLOPS")
            typer.echo(f"     INT8:  {peak.int8_tops:,.1f} TOPS")
            typer.echo("")
            typer.echo("   Memory:")
            typer.echo(f"     Bandwidth: {peak.memory_bandwidth_gbps:,.0f} GB/s")
            typer.echo(f"     Size:      {peak.memory_size_gb:.0f} GB")
            typer.echo(f"     L2 Cache BW: ~{peak.l2_cache_bandwidth_gbps:,.0f} GB/s")
            typer.echo("")
            typer.echo(f"   TDP: {peak.tdp_watts:.0f}W")

    @app.command("quarantine-report")
    def quarantine_report(
        format: str = Option("text", "--format", "-f", help="text, markdown, or json"),
        output: Optional[Path] = Option(None, "--output", "-o", help="Output file"),
        quarantine_file: Optional[Path] = Option(None, "--quarantine-file", help="Path to quarantine.json"),
    ):
        """Generate report of quarantined benchmarks."""
        try:
            from core.scripts.ci.generate_quarantine_report import (
                generate_text_report,
                generate_markdown_report,
                generate_json_report,
            )
            from core.benchmark.quarantine import QuarantineManager
        except ImportError as exc:
            typer.echo(f"Quarantine report generation unavailable: {exc}", err=True)
            raise typer.Exit(code=1)
        
        # Initialize quarantine manager
        if quarantine_file:
            cache_dir = quarantine_file.parent
        else:
            cache_dir = Path("artifacts/verify_cache")
        
        manager = QuarantineManager(cache_dir=cache_dir)
        
        # Generate report
        if format == "text":
            report = generate_text_report(manager)
        elif format == "markdown":
            report = generate_markdown_report(manager)
        elif format == "json":
            report = generate_json_report(manager)
        else:
            typer.echo("Format must be text, markdown, or json", err=True)
            raise typer.Exit(code=1)
        
        # Output
        if output:
            output.write_text(report)
            typer.echo(f"✅ Report written to {output}")
        else:
            typer.echo(report)

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

    @app.command("audit")
    def audit(
        chapter: Optional[str] = Option(None, "--chapter", "-c", help="Specific chapter to audit (e.g., ch10)"),
        lab: Optional[str] = Option(None, "--lab", "-l", help="Specific lab to audit (e.g., decode_optimization)"),
        all_targets: bool = Option(False, "--all", "-a", help="Audit all chapters and labs"),
        verbose: bool = Option(False, "--verbose", "-v", help="Show detailed output"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
    ):
        """Audit benchmark verification compliance.
        
        Uses Python introspection to correctly detect inherited methods
        (unlike grep-based approaches). This gives accurate coverage stats.
        
        Example:
            aisp bench audit --all          # Audit everything
            aisp bench audit -c ch10        # Audit specific chapter
            aisp bench audit -l decode_optimization  # Audit specific lab
        """
        import json as json_lib
        from core.scripts.audit_verification_compliance import (
            audit_directory, print_summary, load_benchmark_class, check_compliance
        )
        
        code_dir = Path(__file__).parent.parent.parent
        
        total_compliant = 0
        total_needs_work = 0
        total_errors = 0
        all_results = {}
        
        # Audit chapters
        if chapter:
            chapters = [chapter]
        elif all_targets or not lab:
            chapters = [f"ch{i:02d}" for i in range(1, 21)]
        else:
            chapters = []
        
        for ch in chapters:
            chapter_dir = code_dir / ch
            if not chapter_dir.exists():
                continue
            
            results = audit_directory(chapter_dir)
            if results:
                if json_output:
                    all_results[ch] = results
                else:
                    c, n, e = print_summary(results, f"{ch.upper()}")
                    total_compliant += c
                    total_needs_work += n
                    total_errors += e
        
        # Audit labs
        if lab:
            labs = [lab]
        elif all_targets or not chapter:
            labs_dir = code_dir / "labs"
            if labs_dir.exists():
                labs = [d.name for d in labs_dir.iterdir() if d.is_dir()]
            else:
                labs = []
        else:
            labs = []
        
        for lb in sorted(labs):
            lab_dir = code_dir / "labs" / lb
            if not lab_dir.exists():
                continue
            
            results = audit_directory(lab_dir)
            if results:
                if json_output:
                    all_results[f"lab:{lb}"] = results
                else:
                    c, n, e = print_summary(results, f"LAB: {lb}")
                    total_compliant += c
                    total_needs_work += n
                    total_errors += e
        
        # Output
        if json_output:
            typer.echo(json_lib.dumps({
                "results": all_results,
                "summary": {
                    "compliant": total_compliant,
                    "needs_work": total_needs_work,
                    "errors": total_errors,
                    "total": total_compliant + total_needs_work + total_errors,
                    "coverage_pct": round(100 * total_compliant / max(1, total_compliant + total_needs_work), 1),
                }
            }, indent=2))
        else:
            typer.echo(f"\n{'='*60}")
            typer.echo("GRAND TOTAL")
            typer.echo(f"{'='*60}")
            typer.echo(f"✅ Compliant: {total_compliant}")
            typer.echo(f"⚠️  Needs work: {total_needs_work}")
            typer.echo(f"❌ Errors: {total_errors}")
            typer.echo(f"Total: {total_compliant + total_needs_work + total_errors}")
            
            coverage = (total_compliant / max(1, total_compliant + total_needs_work)) * 100
            typer.echo(f"\n📊 Coverage: {coverage:.1f}%")


def main():
    """Entry point for CLI."""
    if not TYPER_AVAILABLE:
        print("ERROR: typer is required for CLI. Install with: pip install typer")
        sys.exit(1)

    if app is None:
        print("ERROR: CLI not available")
        sys.exit(1)

    app()
