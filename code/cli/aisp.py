#!/usr/bin/env python3
"""
🚀 aisp - AI System Performance CLI (Typer)

Single entry point that now uses Typer for all category/command routing.
Defaults: no args → launch the benchmark TUI; `bench` is mounted as a subcommand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from enum import Enum
from types import SimpleNamespace
from typing import List, Optional

from tools.plugins.loader import load_plugin_apps
try:
    from common.python.capabilities import has_capability
except Exception:
    def has_capability(name: str) -> bool:
        return False

# Extension flag; updated after plugin discovery
EXT_ENABLED = False

try:
    import typer
except ImportError:  # pragma: no cover - Typer required for the CLI
    typer = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# ENV LOADING (single implementation)
# =============================================================================

def load_env() -> None:
    """Load .env and .env.local files."""
    for env_name in [".env", ".env.local"]:
        env_file = REPO_ROOT / env_name
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if env_name == ".env.local" or key not in os.environ:
                            if key and value:
                                os.environ[key] = value

load_env()


# =============================================================================
# Choice enums (Typer-compatible)
# =============================================================================


class ProfilePreset(str, Enum):
    none = "none"
    minimal = "minimal"
    deep_dive = "deep_dive"
    roofline = "roofline"


class BottleneckMode(str, Enum):
    profile = "profile"
    llm = "llm"
    both = "both"


class GoalChoice(str, Enum):
    throughput = "throughput"
    latency = "latency"
    memory = "memory"


class TargetChoice(str, Enum):
    throughput = "throughput"
    latency = "latency"
    balanced = "balanced"


class ReportFormat(str, Enum):
    markdown = "markdown"
    json = "json"


class ExportFormat(str, Enum):
    json = "json"
    csv = "csv"
    detailed_csv = "detailed_csv"
    html = "html"


class SpeedTestChoice(str, Enum):
    all = "all"
    gemm = "gemm"
    attention = "attention"
    memory = "memory"


# =============================================================================
# Helpers
# =============================================================================

def _typer_required() -> None:
    print("Typer is required for aisp. Install with: pip install typer")
    sys.exit(1)


def _build_args(ctx: typer.Context, **kwargs) -> SimpleNamespace:
    """Inject common flags into a SimpleNamespace for legacy command functions."""
    verbose = False
    json_output = False
    if ctx and ctx.obj:
        verbose = bool(ctx.obj.get("verbose", False))
        json_output = bool(ctx.obj.get("json_output", False))
    normalized = {
        key: (value.value if isinstance(value, Enum) else value) for key, value in kwargs.items()
    }
    return SimpleNamespace(verbose=verbose, json=json_output, **normalized)


def _run(func, ctx: typer.Context, **kwargs) -> None:
    """Execute legacy command functions and honor their integer exit codes."""
    args = _build_args(ctx, **kwargs)
    try:
        result = func(args)
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
    code = 0 if result is None else int(result)
    raise typer.Exit(code=code)


def _rewrite_help_aliases(argv: List[str]) -> List[str]:
    """Support `aisp help` and `aisp <category> help` as help aliases."""
    if len(argv) > 1 and argv[1] == "help":
        return [argv[0], "help", *argv[2:]]
    if len(argv) > 2 and argv[2] == "help":
        return [argv[0], "help", argv[1], *argv[3:]]
    return argv


# =============================================================================
# Typer app + subcommands
# =============================================================================

if typer:
    app = typer.Typer(
        name="aisp",
        help="AI Systems Performance CLI",
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    system_app = typer.Typer(help="System diagnostics, environment, and GPU status")
    test_app = typer.Typer(help="Lightweight tests (bandwidth, network, warmup)")
    microbench_app = typer.Typer(help="Microbenchmarks (disk, PCIe, memory, tensor)")

    # Extension categories (added only if extensions are enabled)
    ai_app = typer.Typer(help="LLM-powered Q&A, explanations, troubleshooting")
    analyze_app = typer.Typer(help="Profile analysis, comparisons, roofline")
    optimize_app = typer.Typer(help="Optimization recommendations and playbooks")
    distributed_app = typer.Typer(help="Parallelism planning, topology, NCCL tuning")
    inference_app = typer.Typer(help="Inference configuration, quantization, deployment")
    training_app = typer.Typer(help="Training helpers (RL, checkpoints, gradients)")
    monitor_app = typer.Typer(help="Monitoring and regression detection")
    report_app = typer.Typer(help="Reports, history, ROI, exports")
    profile_app = typer.Typer(help="Profiling helpers (flame, NCU, HTA, divergence)")
    hf_app = typer.Typer(help="HuggingFace model search and config")
    cluster_app = typer.Typer(help="Cluster scripts, cost, scaling, power")


# =============================================================================
# Root callback
# =============================================================================

if typer:

    @app.callback(invoke_without_command=True)
    def _root(
        ctx: typer.Context,
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON when supported"),
    ) -> None:
        """Default: launch the benchmark TUI when no subcommand is provided."""
        ctx.obj = {"verbose": verbose, "json_output": json_output}
        if ctx.invoked_subcommand is None:
            from tools.cli.tui import run_tui
            try:
                run_tui()
            except Exception as exc:  # pragma: no cover - curses may fail in CI
                typer.echo(f"TUI error: {exc}", err=True)
                raise typer.Exit(code=1)
            raise typer.Exit()


# =============================================================================
# Category: system
# =============================================================================

if typer:

    @system_app.command("status", help="Show comprehensive system status")
    def system_status(ctx: typer.Context) -> None:
        from tools.cli.commands import system as system_cmds
        _run(system_cmds.system_status, ctx)

    @system_app.command("gpu", help="GPU information and control")
    def system_gpu(ctx: typer.Context) -> None:
        from tools.cli.commands import system as system_cmds
        _run(system_cmds.gpu_info, ctx)

    @system_app.command("env", help="Environment variables and paths")
    def system_env(ctx: typer.Context) -> None:
        from tools.cli.commands import system as system_cmds
        _run(system_cmds.show_env, ctx)

    @system_app.command("deps", help="Check dependencies and versions")
    def system_deps(ctx: typer.Context) -> None:
        from tools.cli.commands import system as system_cmds
        _run(system_cmds.check_deps, ctx)

    @system_app.command("preflight", help="Pre-flight checks before optimization")
    def system_preflight(ctx: typer.Context) -> None:
        from tools.cli.commands import system as system_cmds
        _run(system_cmds.preflight, ctx)


# =============================================================================
# Category: ai (extension)
# =============================================================================

if typer and EXT_ENABLED and ai_app is not None:

    @ai_app.command("ask", help="Ask a performance question (LLM-powered)")
    def ai_ask(
        ctx: typer.Context,
        question: Optional[List[str]] = typer.Argument(None, help="Question to ask"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from tools.cli.commands import ai_assistant
        _run(ai_assistant.ask_question, ctx, question=question, no_book=no_book)

    @ai_app.command("explain", help="Explain a concept with book + LLM")
    def ai_explain(
        ctx: typer.Context,
        concept: Optional[str] = typer.Argument(None, help="Concept to explain"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from tools.cli.commands import ai_assistant
        _run(ai_assistant.explain_concept, ctx, concept=concept, no_book=no_book)

    @ai_app.command("troubleshoot", help="Diagnose and fix issues")
    def ai_troubleshoot(
        ctx: typer.Context,
        issue: Optional[List[str]] = typer.Argument(None, help="Issue description"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from tools.cli.commands import ai_assistant
        _run(ai_assistant.troubleshoot, ctx, issue=issue, no_book=no_book)

    @ai_app.command("status", help="Check LLM backend status")
    def ai_status(ctx: typer.Context) -> None:
        from tools.cli.commands import ai_assistant
        _run(ai_assistant.llm_status, ctx)


# =============================================================================
# Category: analyze (extension)
# =============================================================================

if typer and EXT_ENABLED and analyze_app is not None:

    @analyze_app.command("profile", help="Run profiling on a benchmark/script")
    def analyze_profile(
        ctx: typer.Context,
        targets: Optional[List[str]] = typer.Option(
            None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to profile"
        ),
        profile: ProfilePreset = typer.Option(
            ProfilePreset.deep_dive,
            "--profile",
            "-p",
            help="Benchmark CLI profile preset",
            show_choices=True,
        ),
    ) -> None:
        from tools.cli.commands import analysis
        _run(analysis.run_profile, ctx, targets=targets, profile=profile)

    @analyze_app.command("compare", help="Compare two configurations/runs")
    def analyze_compare(
        ctx: typer.Context,
        chapter: str = typer.Option(
            "labs/moe_parallelism",
            "--chapter",
            "-c",
            help="Chapter token (e.g., ch15 or labs/moe_parallelism)",
        ),
        targets: Optional[List[str]] = typer.Option(
            None, "--targets", help="Optional example names to compare"
        ),
    ) -> None:
        from tools.cli.commands import analysis
        _run(analysis.compare_runs, ctx, chapter=chapter, targets=targets)

    @analyze_app.command("diff", help="Differential analysis between baseline/optimized")
    def analyze_diff(
        ctx: typer.Context,
        baseline: Optional[Path] = typer.Argument(None, help="Baseline deep profile JSON"),
        optimized: Optional[Path] = typer.Argument(None, help="Optimized deep profile JSON"),
    ) -> None:
        from tools.cli.commands import analysis
        _run(analysis.diff_analysis, ctx, baseline=str(baseline) if baseline else None, optimized=str(optimized) if optimized else None)

    @analyze_app.command("roofline", help="Roofline model analysis")
    def analyze_roofline(ctx: typer.Context) -> None:
        from tools.cli.commands import analysis
        _run(analysis.roofline, ctx)

    @analyze_app.command("bottleneck", help="Identify performance bottlenecks")
    def analyze_bottleneck(
        ctx: typer.Context,
        mode: BottleneckMode = typer.Option(
            BottleneckMode.both,
            "--mode",
            help="Use profile-only, LLM-only, or combined analysis",
            show_choices=True,
        ),
        limit: int = typer.Option(5, "--limit", help="Limit number of bottlenecks shown"),
    ) -> None:
        from tools.cli.commands import analysis
        _run(analysis.bottleneck, ctx, mode=mode, limit=limit)


# =============================================================================
# Category: optimize (extension)
# =============================================================================

if typer and EXT_ENABLED and optimize_app is not None:

    @optimize_app.command("recommend", help="Get optimization recommendations")
    def optimize_recommend(
        ctx: typer.Context,
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
        goal: GoalChoice = typer.Option(
            GoalChoice.throughput,
            "--goal",
            help="Optimization goal",
            show_choices=True,
        ),
    ) -> None:
        from tools.cli.commands import optimization
        _run(optimization.recommend, ctx, model_size=model_size, gpus=gpus, goal=goal)

    @optimize_app.command("auto", help="Auto-optimization with LLM guidance")
    def optimize_auto(ctx: typer.Context) -> None:
        from tools.cli.commands import optimization
        _run(optimization.auto_optimize, ctx)

    @optimize_app.command("whatif", help="What-if analysis for optimizations")
    def optimize_whatif(ctx: typer.Context) -> None:
        from tools.cli.commands import optimization
        _run(optimization.whatif, ctx)

    @optimize_app.command("stacking", help="Compound optimization stacking")
    def optimize_stacking(ctx: typer.Context) -> None:
        from tools.cli.commands import optimization
        _run(optimization.stacking, ctx)

    @optimize_app.command("playbook", help="Pre-built optimization playbooks")
    def optimize_playbook(
        ctx: typer.Context,
        action: str = typer.Argument("list", help='Playbook name or "list"'),
    ) -> None:
        from tools.cli.commands import optimization
        _run(optimization.playbook, ctx, action=action)


# =============================================================================
# Category: distributed (extension)
# =============================================================================

if typer and EXT_ENABLED and distributed_app is not None:

    @distributed_app.command("plan", help="Plan parallelism strategy (TP/PP/DP)")
    def distributed_plan(
        ctx: typer.Context,
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(8, "--gpus", help="Number of GPUs"),
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
    ) -> None:
        from tools.cli.commands import distributed
        _run(distributed.plan_parallelism, ctx, model_size=model_size, gpus=gpus, nodes=nodes)

    @distributed_app.command("topology", help="Analyze GPU topology")
    def distributed_topology(ctx: typer.Context) -> None:
        from tools.cli.commands import distributed
        _run(distributed.topology, ctx)

    @distributed_app.command("nccl", help="NCCL tuning recommendations")
    def distributed_nccl(
        ctx: typer.Context,
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
        gpus: int = typer.Option(8, "--gpus", help="GPUs per node"),
        diagnose: bool = typer.Option(False, "--diagnose", help="Include diagnostic checks"),
    ) -> None:
        from tools.cli.commands import distributed
        _run(distributed.nccl_tuning, ctx, nodes=nodes, gpus=gpus, diagnose=diagnose)

    @distributed_app.command("zero", help="ZeRO/FSDP configuration")
    def distributed_zero(
        ctx: typer.Context,
        model_size: float = typer.Option(70.0, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(8, "--gpus", help="Number of GPUs"),
    ) -> None:
        from tools.cli.commands import distributed
        _run(distributed.zero_config, ctx, model_size=model_size, gpus=gpus)


# =============================================================================
# Category: inference (extension)
# =============================================================================

if typer and EXT_ENABLED and inference_app is not None:

    @inference_app.command("vllm", help="vLLM configuration and optimization")
    def inference_vllm(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(None, "--model", help="Model name"),
        target: TargetChoice = typer.Option(
            TargetChoice.balanced,
            "--target",
            help="Optimization target",
            show_choices=True,
        ),
    ) -> None:
        from tools.cli.commands import inference
        _run(inference.vllm_config, ctx, model=model, target=target)

    @inference_app.command("quantize", help="Quantization recommendations")
    def inference_quantize(
        ctx: typer.Context,
        model_size: float = typer.Option(70.0, "--model-size", help="Model size in billions"),
        target_memory: Optional[float] = typer.Option(
            None, "--target-memory", help="Target memory per GPU (GB)"
        ),
    ) -> None:
        from tools.cli.commands import inference
        _run(inference.quantize, ctx, model_size=model_size, target_memory=target_memory)

    @inference_app.command("deploy", help="Deployment configuration")
    def inference_deploy(
        ctx: typer.Context,
        model: str = typer.Option("meta-llama/Llama-2-70b-hf", "--model", help="Model name"),
        target: str = typer.Option("vllm", "--target", help="Backend target (vllm/tensorrt/triton)"),
    ) -> None:
        from tools.cli.commands import inference
        _run(inference.deploy_config, ctx, model=model, target=target)

    @inference_app.command("serve", help="Start inference server")
    def inference_serve(
        ctx: typer.Context,
        model: str = typer.Option("meta-llama/Llama-2-70b-hf", "--model", help="Model name"),
        gpus: int = typer.Option(1, "--gpus", help="Tensor parallel size"),
    ) -> None:
        from tools.cli.commands import inference
        _run(inference.serve, ctx, model=model, gpus=gpus)


# =============================================================================
# Category: training (extension)
# =============================================================================

if typer and EXT_ENABLED and training_app is not None:

    @training_app.command("rl", help="RL/RLHF optimization")
    def training_rl(ctx: typer.Context) -> None:
        from tools.cli.commands import training
        _run(training.rl, ctx)

    @training_app.command("checkpoint", help="Checkpointing strategy")
    def training_checkpoint(ctx: typer.Context) -> None:
        from tools.cli.commands import training
        _run(training.checkpointing, ctx)

    @training_app.command("gradient", help="Gradient optimization")
    def training_gradient(ctx: typer.Context) -> None:
        from tools.cli.commands import training
        _run(training.gradient, ctx)


# =============================================================================
# Category: monitor (extension)
# =============================================================================

if typer and EXT_ENABLED and monitor_app is not None:

    @monitor_app.command("live", help="Real-time GPU monitoring")
    def monitor_live(ctx: typer.Context) -> None:
        from tools.cli.commands import monitoring
        _run(monitoring.live_monitor, ctx)

    @monitor_app.command("regression", help="Detect performance regressions")
    def monitor_regression(ctx: typer.Context) -> None:
        from tools.cli.commands import monitoring
        _run(monitoring.regression, ctx)

    @monitor_app.command("metrics", help="Collect and display metrics")
    def monitor_metrics(ctx: typer.Context) -> None:
        from tools.cli.commands import monitoring
        _run(monitoring.metrics, ctx)


# =============================================================================
# Category: report (extension)
# =============================================================================

if typer and EXT_ENABLED and report_app is not None:

    @report_app.command("generate", help="Generate performance report")
    def report_generate(
        ctx: typer.Context,
        output: Optional[Path] = typer.Option(None, "--output", help="Output file for generated report"),
        fmt: ReportFormat = typer.Option(
            ReportFormat.markdown,
            "--format",
            help="Report format",
            show_choices=True,
        ),
    ) -> None:
        from tools.cli.commands import reports
        _run(reports.generate_report, ctx, output=str(output) if output else None, format=fmt)

    @report_app.command("history", help="View optimization history")
    def report_history(ctx: typer.Context) -> None:
        from tools.cli.commands import reports
        _run(reports.show_history, ctx)

    @report_app.command("roi", help="Calculate ROI of optimizations")
    def report_roi(ctx: typer.Context) -> None:
        from tools.cli.commands import reports
        _run(reports.calculate_roi, ctx)

    @report_app.command("export", help="Export results to various formats")
    def report_export(
        ctx: typer.Context,
        fmt: ExportFormat = typer.Option(
            ExportFormat.json,
            "--format",
            help="Export format",
            show_choices=True,
        ),
        output: Optional[Path] = typer.Option(None, "--output", help="Output file"),
    ) -> None:
        from tools.cli.commands import reports
        _run(reports.export, ctx, format=fmt, output=str(output) if output else None)


# =============================================================================
# Category: profile (extension)
# =============================================================================

if typer and EXT_ENABLED and profile_app is not None:

    @profile_app.command("flame", help="Generate flame graph")
    def profile_flame(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
        output: str = typer.Option("flame.html", "--output", "-o", help="Output file"),
    ) -> None:
        from tools.cli.commands import profiling
        _run(profiling.flame, ctx, file=str(file) if file else None, output=output)

    @profile_app.command("memory", help="Memory timeline analysis")
    def profile_memory(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional path to write JSON timeline"),
    ) -> None:
        from tools.cli.commands import profiling
        _run(profiling.memory, ctx, file=str(file) if file else None, output=output)

    @profile_app.command("kernels", help="Kernel breakdown analysis")
    def profile_kernels(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
    ) -> None:
        from tools.cli.commands import profiling
        _run(profiling.kernels, ctx, file=str(file) if file else None)

    @profile_app.command("hta", help="Holistic Trace Analysis")
    def profile_hta(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
    ) -> None:
        from tools.cli.commands import profiling
        _run(profiling.hta, ctx, file=str(file) if file else None)

    @profile_app.command("ncu", help="NCU deep dive analysis")
    def profile_ncu(
        ctx: typer.Context,
        script: Optional[Path] = typer.Argument(None, help="Script to profile"),
        kernel: Optional[str] = typer.Option(None, "--kernel", help="Specific kernel"),
    ) -> None:
        from tools.cli.commands import profiling
        _run(profiling.ncu, ctx, script=str(script) if script else None, kernel=kernel)

    @profile_app.command("warp-divergence", help="Warp divergence analysis")
    def profile_warp_divergence(ctx: typer.Context) -> None:
        from tools.cli.commands import profiling
        _run(profiling.warp_divergence, ctx)

    @profile_app.command("bank-conflicts", help="Bank conflict analysis")
    def profile_bank_conflicts(ctx: typer.Context) -> None:
        from tools.cli.commands import profiling
        _run(profiling.bank_conflicts, ctx)

    @profile_app.command("occupancy", help="Occupancy analysis")
    def profile_occupancy(ctx: typer.Context) -> None:
        from tools.cli.commands import profiling
        _run(profiling.occupancy, ctx)


# =============================================================================
# Category: HuggingFace (extension)
# =============================================================================

if typer and EXT_ENABLED and hf_app is not None:

    @hf_app.command("search", help="Search HuggingFace models")
    def hf_search(
        ctx: typer.Context,
        query: Optional[str] = typer.Argument(None, help="Search query"),
        task: Optional[str] = typer.Option(None, "--task", help="Filter by task"),
        limit: int = typer.Option(10, "--limit", help="Max results"),
    ) -> None:
        from tools.cli.commands import huggingface
        _run(huggingface.search, ctx, query=query, task=task, limit=limit)

    @hf_app.command("trending", help="Trending models")
    def hf_trending(
        ctx: typer.Context,
        task: str = typer.Option("text-generation", "--task", help="Task type"),
        limit: int = typer.Option(15, "--limit", help="Max results"),
    ) -> None:
        from tools.cli.commands import huggingface
        _run(huggingface.trending, ctx, task=task, limit=limit)

    @hf_app.command("model", help="Model information")
    def hf_model(
        ctx: typer.Context,
        model: Optional[str] = typer.Argument(None, help="Model ID"),
    ) -> None:
        from tools.cli.commands import huggingface
        _run(huggingface.model_info, ctx, model=model)

    @hf_app.command("config", help="Generate optimal config")
    def hf_config(
        ctx: typer.Context,
        model: Optional[str] = typer.Argument(None, help="Model ID"),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
    ) -> None:
        from tools.cli.commands import huggingface
        _run(huggingface.model_config, ctx, model=model, gpus=gpus)


# =============================================================================
# Category: test
# =============================================================================

if typer:

    @test_app.command("bandwidth", help="GPU memory bandwidth test")
    def test_bandwidth(ctx: typer.Context) -> None:
        from tools.cli.commands import tests as test_cmds
        _run(test_cmds.gpu_bandwidth, ctx)

    @test_app.command("network", help="Network throughput test")
    def test_network(ctx: typer.Context) -> None:
        from tools.cli.commands import tests as test_cmds
        _run(test_cmds.network_test, ctx)

    @test_app.command("warmup", help="Warmup/JIT audit")
    def test_warmup(
        ctx: typer.Context,
        script: Optional[Path] = typer.Argument(None, help="Script to analyze"),
        iterations: int = typer.Option(10, "--iterations", help="Iterations"),
    ) -> None:
        from tools.cli.commands import tests as test_cmds
        _run(test_cmds.warmup_audit, ctx, script=str(script) if script else None, iterations=iterations)

    @test_app.command("speed", help="Run speed tests")
    def test_speed(
        ctx: typer.Context,
        type: SpeedTestChoice = typer.Option(
            SpeedTestChoice.all,
            "--type",
            help="Speed test type",
            show_choices=True,
        ),
    ) -> None:
        from tools.cli.commands import tests as test_cmds
        _run(test_cmds.speedtest, ctx, type=type)

    @test_app.command("diagnostics", help="System diagnostics")
    def test_diagnostics(ctx: typer.Context) -> None:
        from tools.cli.commands import tests as test_cmds
        _run(test_cmds.diagnostics, ctx)


# =============================================================================
# Category: cluster (extension)
# =============================================================================

if typer and EXT_ENABLED and cluster_app is not None:

    @cluster_app.command("slurm", help="Generate SLURM script")
    def cluster_slurm(
        ctx: typer.Context,
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
        gpus: int = typer.Option(8, "--gpus", help="GPUs per node"),
        model: str = typer.Option("llama-70b", "--model", help="Model name"),
        time: str = typer.Option("24:00:00", "--time", help="Time limit"),
        partition: str = typer.Option("gpu", "--partition", help="SLURM partition"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    ) -> None:
        from tools.cli.commands import cluster
        _run(
            cluster.slurm_generate,
            ctx,
            nodes=nodes,
            gpus=gpus,
            model=model,
            time=time,
            partition=partition,
            output=str(output) if output else None,
        )

    @cluster_app.command("cost", help="Cloud cost calculator")
    def cluster_cost(
        ctx: typer.Context,
        model_size: float = typer.Option(70.0, "--model-size", help="Model size (B)"),
        tokens: float = typer.Option(1e12, "--tokens", help="Training tokens"),
        batch_size: int = typer.Option(1024, "--batch-size", help="Batch size"),
    ) -> None:
        from tools.cli.commands import cluster
        _run(cluster.cloud_cost, ctx, model_size=model_size, tokens=tokens, batch_size=batch_size)

    @cluster_app.command("scaling", help="Predict scaling efficiency")
    def cluster_scaling(
        ctx: typer.Context,
        gpus: int = typer.Option(8, "--gpus", help="Target GPUs"),
        model_size: float = typer.Option(70.0, "--model-size", help="Model size (B)"),
        batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    ) -> None:
        from tools.cli.commands import cluster
        _run(cluster.scaling_predict, ctx, gpus=gpus, model_size=model_size, batch_size=batch_size)

    @cluster_app.command("diagnose", help="Cluster diagnostics")
    def cluster_diagnose(ctx: typer.Context) -> None:
        from tools.cli.commands import cluster
        _run(cluster.cluster_diagnose, ctx)

    @cluster_app.command("power", help="Power analysis")
    def cluster_power(ctx: typer.Context) -> None:
        from tools.cli.commands import cluster
        _run(cluster.power_analysis, ctx)


# =============================================================================
# Category: microbench (core)
# =============================================================================

if typer:

    @microbench_app.command("disk", help="Disk I/O benchmark (sequential)")
    def microbench_disk(
        ctx: typer.Context,
        file_size_mb: int = typer.Option(256, "--size-mb", "-s", help="File size (MB)"),
        block_size_kb: int = typer.Option(1024, "--block-kb", "-b", help="Block size (KB)"),
        tmp_dir: Optional[Path] = typer.Option(None, "--tmp-dir", help="Temporary directory"),
    ) -> None:
        from tools.cli.commands import microbench
        _run(
            microbench.disk,
            ctx,
            file_size_mb=file_size_mb,
            block_size_kb=block_size_kb,
            tmp_dir=str(tmp_dir) if tmp_dir else None,
        )

    @microbench_app.command("pcie", help="PCIe H2D/D2H bandwidth")
    def microbench_pcie(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        iters: int = typer.Option(10, "--iters", "-i", help="Iterations"),
    ) -> None:
        from tools.cli.commands import microbench
        _run(microbench.pcie, ctx, size_mb=size_mb, iters=iters)

    @microbench_app.command("mem", help="Memory hierarchy stride test")
    def microbench_mem(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Buffer size (MB)"),
        stride: int = typer.Option(128, "--stride", help="Stride (bytes)"),
    ) -> None:
        from tools.cli.commands import microbench
        _run(microbench.mem_hierarchy, ctx, size_mb=size_mb, stride=stride)

    @microbench_app.command("tensor", help="Tensor core throughput")
    def microbench_tensor(
        ctx: typer.Context,
        size: int = typer.Option(4096, "--size", help="Matrix size"),
        precision: str = typer.Option("fp16", "--precision", help="Precision (fp16/fp32/bf16)"),
    ) -> None:
        from tools.cli.commands import microbench
        _run(microbench.tensor_core, ctx, size=size, precision=precision)

    @microbench_app.command("sfu", help="SFU benchmark")
    def microbench_sfu(
        ctx: typer.Context,
        elements: int = typer.Option(64 * 1024 * 1024, "--elements", help="Number of elements"),
    ) -> None:
        from tools.cli.commands import microbench
        _run(microbench.sfu, ctx, elements=elements)

    @microbench_app.command("loopback", help="Loopback TCP throughput")
    def microbench_loopback(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        port: int = typer.Option(5789, "--port", help="Port to use"),
    ) -> None:
        from tools.cli.commands import microbench
        _run(microbench.loopback, ctx, size_mb=size_mb, port=port)


# =============================================================================
# Top-level commands (bench, version, help; extensions add more)
# =============================================================================

if typer:

    # Bench integration (core)
    try:
        from tools.cli import bench as bench_cli
        BENCH_APP = bench_cli.app if getattr(bench_cli, "TYPER_AVAILABLE", False) else None
    except Exception:
        BENCH_APP = None

    plugin_apps = load_plugin_apps()
    EXT_ENABLED = bool(os.environ.get("AISP_ENABLE_EXT")) or has_capability("cli.ext") or bool(plugin_apps)

    if BENCH_APP:
        app.add_typer(BENCH_APP, name="bench", help="Run, profile, and verify benchmarks")
    else:  # pragma: no cover - bench missing during docs builds

        @app.command("bench", help="Run, profile, and verify benchmarks")
        def bench_stub() -> None:
            typer.echo("Bench CLI unavailable (typer not installed or import failed).")
            raise typer.Exit(code=1)

    # Extension-owned top-level commands should be provided via plugins
    if EXT_ENABLED:

        @app.command("tui", help="Launch interactive terminal UI")
        def tui_command(
            ctx: typer.Context,
            data_file: Optional[Path] = typer.Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
        ) -> None:
            from tools.cli.tui import run_tui
            try:
                run_tui(str(data_file) if data_file else None)
            except Exception as exc:  # pragma: no cover - curses may fail in CI
                typer.echo(f"TUI error: {exc}", err=True)
                raise typer.Exit(code=1)

        @app.command("dashboard", help="Launch web dashboard")
        def dashboard_command(
            ctx: typer.Context,
            port: int = typer.Option(6970, "--port", "-p", help="Port to run the server on"),
            data: Optional[Path] = typer.Option(None, "--data", "-d", help="Path to benchmark results JSON file"),
            no_browser: bool = typer.Option(False, "--no-browser", help="Do not open browser automatically"),
        ) -> None:
            from tools.dashboard.server import serve_dashboard
            serve_dashboard(port=port, data_file=data, open_browser=not no_browser)

        @app.command("mcp", help="Start MCP server for AI chat integration")
        def mcp_command(
            ctx: typer.Context,
            list_tools: bool = typer.Option(False, "--list", help="List available tools"),
            test: Optional[str] = typer.Option(None, "--test", help="Test a specific tool"),
            serve: bool = typer.Option(False, "--serve", help="Start MCP server (stdio)"),
        ) -> None:
            from tools.mcp import server

            old_argv = sys.argv
            new_argv = [old_argv[0]]
            if list_tools:
                new_argv.append("--list")
            if test:
                new_argv.extend(["--test", test])
            if serve:
                new_argv.append("--serve")
            sys.argv = new_argv
            try:
                result = server.main()
                raise typer.Exit(code=0 if result is None else int(result))
            finally:
                sys.argv = old_argv

    @app.command("version", help="Show version information")
    def version_command() -> None:
        typer.echo("aisp - AI System Performance CLI (Typer)")
        typer.echo(f"Repository: {REPO_ROOT}")

    @app.command("help", help="Show help (alias for --help)")
    def help_command(topic: Optional[List[str]] = typer.Argument(None, help="Command or category to show help for")) -> None:
        click_cmd = typer.main.get_command(app)
        args = ["--help"] if not topic else [*topic, "--help"]
        try:
            click_cmd.main(args=args, prog_name="aisp", standalone_mode=False)
        except SystemExit as exc:
            raise typer.Exit(code=exc.code or 0)

    # Load optional plugin Typer apps via entry points (aipe.plugins)
    for plugin in plugin_apps:
        try:
            app.add_typer(plugin.app, name=plugin.name, help=plugin.description or None)
        except Exception:
            continue


# =============================================================================
# Attach sub-apps
# =============================================================================

if typer:
    app.add_typer(system_app, name="system")
    app.add_typer(test_app, name="test")
    app.add_typer(microbench_app, name="microbench")
    if EXT_ENABLED:
        app.add_typer(ai_app, name="ai")
        app.add_typer(analyze_app, name="analyze")
        app.add_typer(optimize_app, name="optimize")
        app.add_typer(distributed_app, name="distributed")
        app.add_typer(inference_app, name="inference")
        app.add_typer(training_app, name="training")
        app.add_typer(monitor_app, name="monitor")
        app.add_typer(report_app, name="report")
        app.add_typer(profile_app, name="profile")
        app.add_typer(hf_app, name="hf")
        app.add_typer(cluster_app, name="cluster")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> int:
    if typer is None:
        _typer_required()
    rewritten = _rewrite_help_aliases(sys.argv)
    sys.argv = rewritten
    try:
        app()
    except SystemExit as exc:  # Typer raises SystemExit
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
