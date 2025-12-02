#!/usr/bin/env python3
"""
🚀 aisp - AI Systems Performance CLI

Single entry point for the AI Performance Engineering system.
Defaults: no args → launch the benchmark TUI.

ARCHITECTURE:
    This CLI uses the unified PerformanceEngine (10 domains) as its backend.
    Commands are organized to match the engine's domain model:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  CLI Command Groups       →  Engine Domains                          │
    │──────────────────────────────────────────────────────────────────────│
    │  aisp info [status|gpu|deps|features|network|topo]  → gpu, system    │
    │  aisp hw [memory|pcie|tc|...]                       → benchmark      │
    │  aisp profile [nsys|ncu|compare|flame|hta]          → profile        │
    │  aisp optimize [recommend|roi|techniques]           → optimize       │
    │  aisp distributed [plan|nccl|...]                   → distributed    │
    │  aisp inference [vllm|quantize|...]                 → inference      │
    │  aisp ai [ask|explain|troubleshoot]                 → ai             │
    │  aisp report [export|history|roi]                   → export         │
    │  aisp bench [run|targets|compare]                   → benchmark      │
    └──────────────────────────────────────────────────────────────────────┘

DOMAIN MODEL (10 domains in core/engine.py):
    1. gpu        - Hardware info, topology, power, bandwidth
    2. system     - Software stack, dependencies, capabilities
    3. profile    - nsys/ncu profiling, flame graphs, HTA
    4. analyze    - Bottlenecks, pareto, scaling, memory patterns
    5. optimize   - Recommendations, ROI, techniques
    6. distributed- Parallelism planning, NCCL, FSDP
    7. inference  - vLLM config, quantization, deployment
    8. benchmark  - Run & track benchmarks, history
    9. ai         - Ask questions, explain concepts, LLM analysis
   10. export     - CSV, PDF, HTML reports

QUICK START:
    aisp                    # Launch interactive TUI
    aisp info status        # System status overview
    aisp info gpu           # GPU information
    aisp ai ask "Why slow?" # Ask AI about performance
    aisp bench run -t ch07  # Run benchmarks
    
EXAMPLES:
    # Get optimization recommendations for 70B model on 8 GPUs
    aisp optimize recommend --model-size 70 --gpus 8
    
    # Profile with Nsight Systems
    aisp profile nsys python train.py
    
    # Plan distributed training
    aisp distributed plan --model-size 70 --gpus 16 --nodes 2
"""

from __future__ import annotations

import os
import sys
import subprocess
import json
from pathlib import Path
from enum import Enum
from types import SimpleNamespace
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.plugins.loader import load_plugin_apps

try:
    import typer
except ImportError:  # pragma: no cover - Typer required for the CLI
    typer = None  # type: ignore

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


def _run_module(module: str, args: List[str]) -> None:
    """Execute a module as a subprocess and propagate its exit code."""
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(cmd)
    raise typer.Exit(code=result.returncode)


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
        help="AI Systems Performance CLI - Unified 10-Domain API",
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    # ==========================================================================
    # 10-DOMAIN COMMAND GROUPS (matching engine.py domains)
    # ==========================================================================
    
    # Domain 1: GPU - Hardware info, topology, power, bandwidth
    gpu_app = typer.Typer(help="GPU hardware: info, topology, power, bandwidth")
    
    # Domain 2: System - Software stack, dependencies, capabilities
    system_app = typer.Typer(help="System: software versions, dependencies, capabilities")
    
    # Domain 3: Profile - nsys/ncu profiling, flame graphs, HTA
    profile_app = typer.Typer(help="Profiling: nsys, ncu, torch, flame graphs, HTA")
    
    # Domain 4: Analyze - Bottlenecks, pareto, scaling, what-if
    analyze_app = typer.Typer(help="Analysis: bottlenecks, pareto, scaling, what-if")
    
    # Domain 5: Optimize - Recommendations, ROI, techniques
    optimize_app = typer.Typer(help="Optimization: recommendations, ROI, techniques")
    
    # Domain 6: Distributed - Parallelism, NCCL, FSDP
    distributed_app = typer.Typer(help="Distributed: parallelism planning, NCCL, FSDP")
    
    # Domain 7: Inference - vLLM, quantization, deployment
    inference_app = typer.Typer(help="Inference: vLLM config, quantization, deployment")
    
    # Domain 8: Benchmark - Run benchmarks, targets, history
    benchmark_app = typer.Typer(help="Benchmarks: run, targets, history, speed tests")
    
    # Domain 9: AI - Ask questions, explain concepts
    ai_app = typer.Typer(help="AI: ask questions, explain concepts, suggest tools")
    
    # Domain 10: Export - CSV, PDF, HTML reports
    export_app = typer.Typer(help="Export: CSV, PDF, HTML reports")
    
    # ==========================================================================
    # LEGACY ALIASES (for backwards compatibility during transition)
    # ==========================================================================
    info_app = gpu_app  # Alias: aisp info → aisp gpu
    hw_app = benchmark_app  # Alias: aisp hw → aisp benchmark  
    report_app = export_app  # Alias: aisp report → aisp export
    
    # Additional legacy groups (will be phased out)
    monitoring_app = typer.Typer(help="[Legacy] Monitoring and diagnostics")
    training_app = typer.Typer(help="[Legacy] Training helpers")
    hf_app = typer.Typer(help="[Legacy] HuggingFace model search")
    cluster_app = typer.Typer(help="[Legacy] Cluster scripts - use 'distributed' instead")


# =============================================================================
# Root callback
# =============================================================================

if typer:

    @app.callback(invoke_without_command=True)
    def _root(
        ctx: typer.Context,
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON when supported"),
        dynamo_logs: bool = typer.Option(
            False,
            "--dynamo-logs/--no-dynamo-logs",
            help="Enable TorchDynamo/TORCH_LOGS output (disabled by default for cleaner CLI)",
        ),
    ) -> None:
        """Default: launch the benchmark TUI when no subcommand is provided."""
        ctx.obj = {"verbose": verbose, "json_output": json_output}
        if not dynamo_logs:
            for key in ["TORCH_LOGS", "TORCH_COMPILE_DEBUG"]:
                os.environ.pop(key, None)
        if ctx.invoked_subcommand is None:
            from cli.tui import run_tui
            try:
                run_tui()
            except Exception as exc:  # pragma: no cover - curses may fail in CI
                typer.echo(f"TUI error: {exc}", err=True)
                raise typer.Exit(code=1)
            raise typer.Exit()


# =============================================================================
# Domain 1: GPU - Hardware info, topology, power, bandwidth
# =============================================================================

if typer:

    @gpu_app.command("info", help="GPU information: name, memory, temperature, utilization")
    def gpu_info_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().gpu.info()
        if ctx.obj and ctx.obj.get("json_output"):
            typer.echo(json.dumps(result, indent=2))
        else:
            typer.echo(f"\n🖥️  GPU Info")
            for i, gpu in enumerate(result.get("gpus", [result])):
                name = gpu.get("name", "Unknown")
                mem = gpu.get("memory_total_gb", 0)
                temp = gpu.get("temperature_c", 0)
                util = gpu.get("utilization_pct", 0)
                typer.echo(f"  GPU {i}: {name}")
                typer.echo(f"    Memory: {mem:.1f} GB, Temp: {temp}°C, Util: {util}%")
        raise typer.Exit(0)

    @gpu_app.command("topology", help="GPU topology: NVLink, PCIe, P2P matrix")
    def gpu_topology_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().gpu.topology()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @gpu_app.command("power", help="Power draw, limits, thermal status")
    def gpu_power_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.power()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @gpu_app.command("bandwidth", help="Run GPU memory bandwidth test")
    def gpu_bandwidth_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        typer.echo("Running GPU bandwidth test...")
        result = get_engine().gpu.bandwidth()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @gpu_app.command("status", help="[Alias] Comprehensive system status")
    def gpu_status(ctx: typer.Context) -> None:
        from cli.commands import system as system_cmds
        _run(system_cmds.system_status, ctx)

    @gpu_app.command("features", help="Hardware features (TMA, tensor cores, FP8)")
    def gpu_features(ctx: typer.Context) -> None:
        """Display detailed hardware capabilities and features."""
        from core.harness.hardware_capabilities import detect_capabilities
        
        cap = detect_capabilities()
        if cap is None:
            typer.echo("❌ CUDA not available on this system.")
            raise typer.Exit(code=1)
        
        typer.echo(f"\n🖥️  GPU: {cap.device_name}")
        typer.echo(f"   Architecture: {cap.name} (SM {cap.sm_version})")
        typer.echo(f"   Compute Capability: {cap.compute_capability}")
        typer.echo(f"   Memory: {cap.total_memory_gb:.1f} GB")
        typer.echo(f"   SMs: {cap.num_sms}")
        typer.echo()
        
        typer.echo("Features:")
        typer.echo(f"  • Tensor Cores: {cap.tensor_cores or 'None'}")
        typer.echo(f"  • TMA: {'✓' if cap.tma_supported else '✗'}")
        typer.echo(f"  • FP8: {'✓' if 'FP8' in cap.features or cap.architecture.lower() in ['hopper', 'blackwell'] else '✗'}")
        typer.echo(f"  • NVLink C2C: {'✓' if cap.nvlink_c2c else '✗'}")
        raise typer.Exit(0)

# =============================================================================
# Domain 2: System - Software stack, dependencies, capabilities  
# =============================================================================

if typer:

    @system_app.command("software", help="Software versions: PyTorch, CUDA, Python")
    def system_software_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.software()
        if ctx.obj and ctx.obj.get("json_output"):
            typer.echo(json.dumps(result, indent=2))
        else:
            typer.echo("\n📦 Software Stack")
            for key, val in result.items():
                typer.echo(f"  {key}: {val}")
        raise typer.Exit(0)

    @system_app.command("deps", help="Check dependency health")
    def system_deps_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.dependencies()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @system_app.command("capabilities", help="Hardware capabilities (TMA, FP8, etc)")
    def system_capabilities_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.capabilities()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @system_app.command("context", help="Full system context for AI analysis")
    def system_context_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.context()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @system_app.command("env", help="Environment variables and paths")
    def system_env_cmd(ctx: typer.Context) -> None:
        from cli.commands import system as system_cmds
        _run(system_cmds.show_env, ctx)

# =============================================================================
# Domain 4: Analyze - Bottlenecks, pareto, scaling, what-if
# =============================================================================

if typer:

    @analyze_app.command("bottlenecks", help="Identify performance bottlenecks")
    def analyze_bottlenecks_cmd(
        ctx: typer.Context,
        mode: BottleneckMode = typer.Option(BottleneckMode.both, "--mode", "-m", help="Analysis mode")
    ) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.bottlenecks(mode=mode.value)
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @analyze_app.command("pareto", help="Pareto frontier analysis")
    def analyze_pareto_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.pareto()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @analyze_app.command("scaling", help="Scaling analysis with GPU count")
    def analyze_scaling_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.scaling()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @analyze_app.command("whatif", help="What-if constraint analysis")
    def analyze_whatif_cmd(
        ctx: typer.Context,
        max_vram: Optional[float] = typer.Option(None, "--max-vram", help="Max VRAM in GB"),
        max_latency: Optional[float] = typer.Option(None, "--max-latency", help="Max latency in ms"),
    ) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.whatif(max_vram_gb=max_vram, max_latency_ms=max_latency)
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    @analyze_app.command("stacking", help="Optimization stacking compatibility")
    def analyze_stacking_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.stacking()
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

# =============================================================================
# Legacy: info (aliased to gpu)
# =============================================================================

if typer:

    @info_app.command("topo", help="Show GPU/NVLink/PCIe topology (nvidia-smi topo -m)")
    def info_topo(ctx: typer.Context) -> None:
        from cli.commands import system as system_cmds
        _run(system_cmds.topo, ctx)

    @info_app.command("check", help="Pre-flight checks before optimization")
    def info_check(ctx: typer.Context) -> None:
        from cli.commands import system as system_cmds
        _run(system_cmds.preflight, ctx)

    @info_app.command("features", help="Hardware features (TMA, TMEM/MMA, DSMEM, tensor cores, FP8, NVLink)")
    def info_features(ctx: typer.Context) -> None:
        """Display detailed hardware capabilities and features."""
        from core.harness.hardware_capabilities import detect_capabilities, format_capability_report
        
        cap = detect_capabilities()
        if cap is None:
            typer.echo("❌ CUDA not available on this system.")
            raise typer.Exit(code=1)
        
        # Header
        typer.echo(f"\n🖥️  GPU: {cap.device_name}")
        typer.echo(f"   Architecture: {cap.name} ({cap.sm_version})")
        typer.echo(f"   Compute Capability: {cap.compute_capability}")
        typer.echo(f"   Memory: {cap.total_memory_gb:.1f} GB", nl=False)
        if cap.memory_bandwidth_tbps:
            typer.echo(f" @ {cap.memory_bandwidth_tbps:.2f} TB/s")
        else:
            typer.echo()
        typer.echo(f"   SMs: {cap.num_sms}, Warp Size: {cap.warp_size}")
        typer.echo()
        
        # Features table
        typer.echo("┌─────────────────────┬────────────────────────────────────┐")
        typer.echo("│ Feature             │ Status                             │")
        typer.echo("├─────────────────────┼────────────────────────────────────┤")
        
        # Tensor Cores
        tc_status = cap.tensor_cores if cap.tensor_cores else "None"
        typer.echo(f"│ Tensor Cores        │ {tc_status:<34} │")
        
        # TMA
        if cap.tma_supported:
            tma_status = f"✓ (1D:{cap.tma_limits.max_1d_elements}, 2D:{cap.tma_limits.max_2d_width}×{cap.tma_limits.max_2d_height})"
            if not cap.tma_compiler_supported:
                tma_status += " [compiler N/A]"
        else:
            tma_status = "✗"
        typer.echo(f"│ TMA                 │ {tma_status:<34} │")
        
        # TMEM (Tensor Memory for MMA accumulators - SM100+ Blackwell)
        tmem_status = "✓" if cap.architecture.lower() in ["blackwell", "blackwell_ultra"] else "✗"
        typer.echo(f"│ TMEM (MMA accum)    │ {tmem_status:<34} │")
        
        # DSMEM (Distributed Shared Memory - cluster feature)
        if cap.cluster.supports_clusters:
            dsmem_status = f"✓ (max cluster: {cap.cluster.max_cluster_size})"
            if cap.cluster.has_dsmem:
                dsmem_status = f"✓ (cluster: {cap.cluster.max_cluster_size})"
        else:
            dsmem_status = "✗"
        typer.echo(f"│ DSMEM (clusters)    │ {dsmem_status:<34} │")
        
        # NVLink
        nvlink_status = "✓" if cap.nvlink_c2c else "✗"
        typer.echo(f"│ NVLink C2C          │ {nvlink_status:<34} │")
        
        # Grace Coherence
        grace_status = "✓" if cap.grace_coherence else "✗"
        typer.echo(f"│ Grace Coherence     │ {grace_status:<34} │")
        
        # Features from list
        fp8 = "✓" if "FP8" in cap.features or cap.architecture.lower() in ["hopper", "blackwell"] else "✗"
        typer.echo(f"│ FP8                 │ {fp8:<34} │")
        
        hbm3e = "✓ HBM3e" if "HBM3e" in cap.features else "✗"
        typer.echo(f"│ HBM3e Memory        │ {hbm3e:<34} │")
        
        # ILP-related info
        typer.echo("├─────────────────────┼────────────────────────────────────┤")
        typer.echo(f"│ Max Threads/Block   │ {cap.max_threads_per_block:<34} │")
        typer.echo(f"│ Max Threads/SM      │ {cap.max_threads_per_sm:<34} │")
        smem_kb = cap.max_shared_mem_per_block / 1024
        typer.echo(f"│ Shared Mem/Block    │ {smem_kb:.0f} KB{'':<28} │")
        if cap.l2_cache_kb:
            typer.echo(f"│ L2 Cache            │ {cap.l2_cache_kb:.0f} KB{'':<28} │")
        
        typer.echo("└─────────────────────┴────────────────────────────────────┘")
        
        # Driver/Runtime info
        if cap.driver_version or cap.cuda_runtime_version:
            typer.echo()
            if cap.driver_version:
                typer.echo(f"   Driver: {cap.driver_version}")
            if cap.cuda_runtime_version:
                typer.echo(f"   CUDA Runtime: {cap.cuda_runtime_version}")
        
        # Notes
        if cap.notes:
            typer.echo("\n📝 Notes:")
            for note in cap.notes:
                typer.echo(f"   • {note}")
        
        typer.echo()

    @info_app.command("network", help="Network and InfiniBand status")
    def info_network(ctx: typer.Context) -> None:
        """Display network interfaces, InfiniBand status, and GPUDirect RDMA info."""
        import shutil
        
        typer.echo("\n🌐 Network Status\n")
        
        # InfiniBand status
        ibstat_cmd = shutil.which("ibstat")
        if ibstat_cmd:
            typer.echo("InfiniBand Devices:")
            try:
                result = subprocess.run([ibstat_cmd], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    # Parse ibstat output for key info
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if any(x in line for x in ['CA ', 'Port ', 'State:', 'Rate:', 'Link layer:']):
                            typer.echo(f"  {line.strip()}")
                else:
                    typer.echo("  No InfiniBand devices found")
            except Exception as e:
                typer.echo(f"  Error querying ibstat: {e}")
        else:
            typer.echo("InfiniBand: ibstat not found (IB tools not installed)")
        
        typer.echo()
        
        # Check for GPUDirect RDMA
        typer.echo("GPUDirect RDMA:")
        try:
            # Check for nv_peer_mem or nvidia_peermem module
            result = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
            if "nv_peer_mem" in result.stdout or "nvidia_peermem" in result.stdout:
                typer.echo("  ✓ Peer memory module loaded (GPUDirect RDMA available)")
            else:
                typer.echo("  ✗ Peer memory module not loaded")
        except Exception:
            typer.echo("  Unable to check peer memory status")
        
        typer.echo()
        
        # NCCL network info
        typer.echo("NCCL Network Environment:")
        nccl_vars = ["NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL", "NCCL_P2P_LEVEL", 
                     "NCCL_IB_GID_INDEX", "NCCL_SOCKET_IFNAME"]
        for var in nccl_vars:
            val = os.environ.get(var, "(not set)")
            typer.echo(f"  {var}: {val}")
        
        typer.echo()


# =============================================================================
# Category: ai
# =============================================================================

if typer and ai_app is not None:

    @ai_app.command("ask", help="Ask a performance question (LLM-powered)")
    def ai_ask(
        ctx: typer.Context,
        question: Optional[List[str]] = typer.Argument(None, help="Question to ask"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.ask_question, ctx, question=question, no_book=no_book)

    @ai_app.command("explain", help="Explain a concept with book + LLM")
    def ai_explain(
        ctx: typer.Context,
        concept: Optional[str] = typer.Argument(None, help="Concept to explain"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.explain_concept, ctx, concept=concept, no_book=no_book)

    @ai_app.command("troubleshoot", help="Diagnose and fix issues")
    def ai_troubleshoot(
        ctx: typer.Context,
        issue: Optional[List[str]] = typer.Argument(None, help="Issue description"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.troubleshoot, ctx, issue=issue, no_book=no_book)

    @ai_app.command("status", help="Check LLM backend status")
    def ai_status(ctx: typer.Context) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.llm_status, ctx)


# =============================================================================
# Category: profile (merged with analyze)
# =============================================================================

if typer and profile_app is not None:

    # --- Capture commands ---
    
    @profile_app.command("nsys", help="Run Nsight Systems capture")
    def profile_nsys(
        ctx: typer.Context,
        command: str = typer.Argument(..., help='Command to profile (quote it), e.g., "python train.py"'),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Base name for output"),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output directory"),
        preset: str = typer.Option("full", "--preset", help="Preset: full or light"),
    ) -> None:
        """Run Nsight Systems profiling on a command."""
        import shlex
        cmd_parts = shlex.split(command)
        output = output_name or "profile"
        out_dir = str(output_dir) if output_dir else "artifacts/profiles"
        
        nsys_cmd = [
            "nsys", "profile",
            "-o", f"{out_dir}/{output}",
            "--trace", "cuda,nvtx,osrt",
            "--cuda-memory-usage=true",
            *cmd_parts
        ]
        typer.echo(f"Running: {' '.join(nsys_cmd)}")
        result = subprocess.run(nsys_cmd)
        raise typer.Exit(code=result.returncode)

    @profile_app.command("ncu", help="Run Nsight Compute capture")
    def profile_ncu(
        ctx: typer.Context,
        script: Optional[Path] = typer.Argument(None, help="Script to profile"),
        kernel: Optional[str] = typer.Option(None, "--kernel", help="Specific kernel filter"),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Output name"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.ncu, ctx, script=str(script) if script else None, kernel=kernel)

    @profile_app.command("torch", help="Run torch.profiler capture (Chrome trace + summary)")
    def profile_torch_profiler(
        ctx: typer.Context,
        script: Path = typer.Argument(..., help="Python script to profile"),
        mode: str = typer.Option("full", "--mode", help="Profiler preset (full/memory/flops/modules/blackwell)"),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Base name for capture folder"),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output root (default: artifacts/torch-profiles)"),
        nvtx_label: str = typer.Option("aisp_torch_profile", "--nvtx-label", help="NVTX/record_function label for correlation"),
        no_nvtx: bool = typer.Option(False, "--no-nvtx", help="Disable NVTX range emission"),
        force_lineinfo: bool = typer.Option(True, "--force-lineinfo/--no-force-lineinfo", help="Force -lineinfo for better source mapping"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
        script_args: List[str] = typer.Option([], "--arg", "-a", help="Arguments forwarded to the script", show_default=False),
    ) -> None:
        from cli.commands import profiling
        _run(
            profiling.torch_profiler,
            ctx,
            script=str(script),
            mode=mode,
            output_name=output_name,
            output_dir=str(output_dir) if output_dir else None,
            nvtx_label=nvtx_label,
            use_nvtx=not no_nvtx,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
            script_args=script_args,
        )

    @profile_app.command("hta-capture", help="Launch HTA capture (nsys + HTAAnalyzer)")
    def profile_hta_capture(
        ctx: typer.Context,
        command: str = typer.Argument(..., help='Command to profile (quote it), e.g., "python train.py --batch 8"'),
        preset: str = typer.Option("full", "--preset", help="Nsight preset (full|light)", show_default=True),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Base name for capture"),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory for nsys + HTA outputs"),
        force_lineinfo: bool = typer.Option(True, "--force-lineinfo/--no-force-lineinfo", help="Force -lineinfo for source mapping"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
    ) -> None:
        from cli.commands import profiling
        _run(
            profiling.hta_capture,
            ctx,
            command=command,
            preset=preset,
            output_name=output_name or "hta_capture",
            output_dir=str(output_dir) if output_dir else None,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
        )

    # --- View commands ---

    @profile_app.command("flame", help="Generate flame graph")
    def profile_flame(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
        output: str = typer.Option("flame.html", "--output", "-o", help="Output file"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.flame, ctx, file=str(file) if file else None, output=output)

    @profile_app.command("memory", help="Memory timeline analysis")
    def profile_memory(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional path to write JSON timeline"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.memory, ctx, file=str(file) if file else None, output=output)

    @profile_app.command("kernels", help="Kernel breakdown analysis")
    def profile_kernels(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.kernels, ctx, file=str(file) if file else None)

    @profile_app.command("hta", help="Holistic Trace Analysis")
    def profile_hta(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.hta, ctx, file=str(file) if file else None)

    # --- Compare commands ---

    @profile_app.command("compare", help="Compare baseline vs optimized profiles (flame graph)")
    def profile_compare(
        ctx: typer.Context,
        chapter: Optional[str] = typer.Argument(None, help="Chapter name or path to profile directory"),
        output: str = typer.Option("comparison_flamegraph.html", "--output", "-o", help="Output HTML file"),
        json_out: Optional[str] = typer.Option(None, "--json", "-j", help="Also output JSON data"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.compare_profiles, ctx, chapter=chapter, output=output, json_out=json_out)

    @profile_app.command("diff", help="Differential analysis between baseline/optimized")
    def profile_diff(
        ctx: typer.Context,
        baseline: Optional[Path] = typer.Argument(None, help="Baseline deep profile JSON"),
        optimized: Optional[Path] = typer.Argument(None, help="Optimized deep profile JSON"),
    ) -> None:
        from cli.commands import analysis
        _run(analysis.diff_analysis, ctx, baseline=str(baseline) if baseline else None, optimized=str(optimized) if optimized else None)

    # --- Analysis commands (merged from analyze) ---

    @profile_app.command("run", help="Run profiling on a benchmark/script")
    def profile_run(
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
        from cli.commands import analysis
        _run(analysis.run_profile, ctx, targets=targets, profile=profile)

    @profile_app.command("roofline", help="Roofline model analysis")
    def profile_roofline(ctx: typer.Context) -> None:
        from cli.commands import analysis
        _run(analysis.roofline, ctx)

    @profile_app.command("bottleneck", help="Identify performance bottlenecks")
    def profile_bottleneck(
        ctx: typer.Context,
        mode: BottleneckMode = typer.Option(
            BottleneckMode.both,
            "--mode",
            help="Use profile-only, LLM-only, or combined analysis",
            show_choices=True,
        ),
        limit: int = typer.Option(5, "--limit", help="Limit number of bottlenecks shown"),
    ) -> None:
        from cli.commands import analysis
        _run(analysis.bottleneck, ctx, mode=mode, limit=limit)

    # --- Advanced analysis commands ---

    @profile_app.command("warp-divergence", help="Warp divergence analysis")
    def profile_warp_divergence(ctx: typer.Context) -> None:
        from cli.commands import profiling
        _run(profiling.warp_divergence, ctx)

    @profile_app.command("bank-conflicts", help="Bank conflict analysis")
    def profile_bank_conflicts(ctx: typer.Context) -> None:
        from cli.commands import profiling
        _run(profiling.bank_conflicts, ctx)

    @profile_app.command("occupancy", help="Occupancy analysis")
    def profile_occupancy(ctx: typer.Context) -> None:
        from cli.commands import profiling
        _run(profiling.occupancy, ctx)

    @profile_app.command("ilp", help="Instruction-level parallelism analysis")
    def profile_ilp(
        ctx: typer.Context,
        report: Optional[Path] = typer.Argument(None, help="NCU report file (.ncu-rep)"),
    ) -> None:
        """Analyze ILP metrics from NCU report or run fresh capture."""
        typer.echo("\n📊 ILP (Instruction-Level Parallelism) Analysis\n")
        
        if report and report.exists():
            # Parse existing NCU report for ILP metrics
            typer.echo(f"Analyzing: {report}")
            # Would use ncu --import to extract metrics
            typer.echo("\nKey ILP Metrics:")
            typer.echo("  • Issued IPC: Instructions per cycle issued")
            typer.echo("  • Executed IPC: Instructions per cycle executed")
            typer.echo("  • Warp cycles per issued instruction")
            typer.echo("  • Pipeline stall reasons")
            typer.echo("\nRun `aisp profile stalls` for detailed stall breakdown.")
        else:
            typer.echo("Usage: aisp profile ilp <ncu-report.ncu-rep>")
            typer.echo("\nTo capture ILP metrics:")
            typer.echo("  ncu --set full --metrics \\")
            typer.echo("    sm__inst_executed_pipe_*,sm__warps_active,sm__inst_issued \\")
            typer.echo("    -o ilp_report python your_script.py")

    @profile_app.command("stalls", help="Pipeline stall analysis")
    def profile_stalls(
        ctx: typer.Context,
        report: Optional[Path] = typer.Argument(None, help="NCU report file (.ncu-rep)"),
    ) -> None:
        """Analyze pipeline stall reasons from NCU report."""
        typer.echo("\n⏱️  Pipeline Stall Analysis\n")
        
        typer.echo("Stall Categories:")
        typer.echo("  • Memory Dependency: Waiting for memory operations")
        typer.echo("  • Execution Dependency: Waiting for ALU/FPU results")
        typer.echo("  • Synchronization: Barrier or atomic waits")
        typer.echo("  • Texture: Waiting for texture fetch")
        typer.echo("  • Pipe Busy: Execution units occupied")
        typer.echo("  • Not Selected: Warp not scheduled (low priority)")
        typer.echo()
        
        if report and report.exists():
            typer.echo(f"Analyzing: {report}")
            # Would parse NCU report for stall metrics
        else:
            typer.echo("To capture stall metrics:")
            typer.echo("  ncu --set full --section WarpStateStats \\")
            typer.echo("    -o stall_report python your_script.py")

    @profile_app.command("warmup", help="Warmup/JIT audit")
    def profile_warmup(
        ctx: typer.Context,
        script: Optional[Path] = typer.Argument(None, help="Script to analyze"),
        iterations: int = typer.Option(10, "--iterations", help="Iterations"),
    ) -> None:
        from cli.commands import tests as test_cmds
        _run(test_cmds.warmup_audit, ctx, script=str(script) if script else None, iterations=iterations)

    @profile_app.command("registers", help="Register pressure and spilling analysis")
    def profile_registers(
        ctx: typer.Context,
        report: Optional[Path] = typer.Argument(None, help="NCU report file (.ncu-rep)"),
    ) -> None:
        """Analyze register usage and spilling from NCU report."""
        typer.echo("\n📊 Register Pressure Analysis\n")
        
        typer.echo("Key Metrics:")
        typer.echo("  • Registers per thread: Target <64 for good occupancy")
        typer.echo("  • Register spills: Local memory accesses (slow)")
        typer.echo("  • Occupancy impact: Fewer registers = more warps")
        typer.echo()
        
        if report and report.exists():
            typer.echo(f"Analyzing: {report}")
            typer.echo("\nTo extract register metrics from NCU:")
            typer.echo("  ncu --query-metrics-mode all | grep -i register")
        else:
            typer.echo("Usage: aisp profile registers <ncu-report.ncu-rep>")
            typer.echo("\nTo capture register metrics:")
            typer.echo("  ncu --set full --section LaunchStats \\")
            typer.echo("    -o register_report python your_script.py")
            typer.echo("\nKey NCU metrics:")
            typer.echo("  • launch__registers_per_thread")
            typer.echo("  • launch__registers_per_thread_allocated")
            typer.echo("  • lts__t_sectors_srcunit_tex_op_read (spill loads)")
            typer.echo("  • lts__t_sectors_srcunit_tex_op_write (spill stores)")

    @profile_app.command("coalescing", help="Memory access coalescing analysis")
    def profile_coalescing(
        ctx: typer.Context,
        stride: int = typer.Option(1, "--stride", help="Access stride in elements"),
        element_size: int = typer.Option(4, "--element-size", help="Element size in bytes"),
    ) -> None:
        """Analyze memory access patterns for coalescing efficiency."""
        typer.echo("\n🔄 Memory Coalescing Analysis\n")
        
        # Calculate coalescing efficiency
        warp_size = 32
        cache_line = 128  # L1/L2 cache line size
        
        access_span = stride * element_size * warp_size
        transactions = (access_span + cache_line - 1) // cache_line
        ideal_transactions = (warp_size * element_size + cache_line - 1) // cache_line
        efficiency = (ideal_transactions / transactions) * 100 if transactions > 0 else 0
        
        typer.echo(f"Configuration:")
        typer.echo(f"  Stride: {stride} elements")
        typer.echo(f"  Element size: {element_size} bytes")
        typer.echo(f"  Warp size: {warp_size} threads")
        typer.echo()
        typer.echo(f"Memory Access Pattern:")
        typer.echo(f"  Access span per warp: {access_span} bytes")
        typer.echo(f"  Cache transactions: {transactions} (ideal: {ideal_transactions})")
        typer.echo(f"  Coalescing efficiency: {efficiency:.1f}%")
        typer.echo()
        
        if efficiency < 50:
            typer.echo("⚠️  Poor coalescing! Consider:")
            typer.echo("  • Transpose data for contiguous access")
            typer.echo("  • Use Structure of Arrays (SoA) instead of AoS")
            typer.echo("  • Pad arrays to avoid bank conflicts")
        elif efficiency < 80:
            typer.echo("⚡ Moderate coalescing. Room for improvement.")
        else:
            typer.echo("✅ Good coalescing efficiency!")

    @profile_app.command("comm-overlap", help="Communication/compute overlap analysis")
    def profile_comm_overlap(
        ctx: typer.Context,
        model: str = typer.Option("llama-3.1-70b", "--model", "-m", help="Model name"),
    ) -> None:
        """Analyze communication/computation overlap opportunities."""
        from core.perf_core_base import PerformanceCore
        
        core = PerformanceCore()
        result = core.get_comm_overlap_analysis(model)
        
        if result.get("success"):
            typer.echo(f"\n🔄 Communication Overlap Analysis: {result.get('model', model)}\n")
            
            inputs = result.get("inputs", {})
            typer.echo(f"Configuration:")
            typer.echo(f"  Model: {inputs.get('params_b', '?')}B params")
            typer.echo(f"  Layout: TP={inputs.get('tp')}, PP={inputs.get('pp')}, DP={inputs.get('dp')}")
            typer.echo(f"  GPUs: {inputs.get('gpus')}")
            typer.echo()
            
            overlap = result.get("overlap_analysis", {})
            if overlap:
                typer.echo(f"Overlap Opportunities:")
                for key, value in overlap.items():
                    if isinstance(value, (int, float)):
                        typer.echo(f"  {key}: {value}")
                    elif isinstance(value, dict):
                        typer.echo(f"  {key}:")
                        for k, v in value.items():
                            typer.echo(f"    {k}: {v}")
            
            if result.get("warning"):
                typer.echo(f"\n⚠️  {result['warning']}")
        else:
            typer.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    @profile_app.command("energy", help="Energy efficiency analysis")
    def profile_energy(ctx: typer.Context) -> None:
        """Analyze GPU energy efficiency and power characteristics."""
        from core.perf_core_base import PerformanceCore
        
        core = PerformanceCore()
        result = core.get_energy_analysis()
        
        if result.get("success"):
            typer.echo("\n⚡ Energy Efficiency Analysis\n")
            
            current = result.get("current_state", {})
            if current:
                typer.echo(f"Current Power State:")
                typer.echo(f"  Power draw: {current.get('power_w', '?')} W")
                typer.echo(f"  Temperature: {current.get('temp_c', '?')}°C")
                typer.echo(f"  GPU utilization: {current.get('gpu_util_pct', '?')}%")
                typer.echo(f"  Memory utilization: {current.get('mem_util_pct', '?')}%")
            
            efficiency = result.get("efficiency_metrics", {})
            if efficiency:
                typer.echo(f"\nEfficiency Metrics:")
                for key, value in efficiency.items():
                    typer.echo(f"  {key}: {value}")
            
            recommendations = result.get("recommendations", [])
            if recommendations:
                typer.echo(f"\n💡 Recommendations:")
                for rec in recommendations:
                    typer.echo(f"  • {rec}")
        else:
            typer.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    @profile_app.command("data-loading", help="Data loading pipeline analysis")
    def profile_data_loading(ctx: typer.Context) -> None:
        """Analyze data loading pipeline for bottlenecks."""
        from core.perf_core_base import PerformanceCore
        
        core = PerformanceCore()
        result = core.get_data_loading_analysis()
        
        if result.get("success"):
            typer.echo("\n📦 Data Loading Analysis\n")
            
            config = result.get("config", {})
            if config:
                typer.echo(f"Current Configuration:")
                for key, value in config.items():
                    typer.echo(f"  {key}: {value}")
            
            bottlenecks = result.get("bottlenecks", [])
            if bottlenecks:
                typer.echo(f"\n⚠️  Bottlenecks Detected:")
                for b in bottlenecks:
                    typer.echo(f"  • {b}")
            
            recommendations = result.get("recommendations", [])
            if recommendations:
                typer.echo(f"\n💡 Recommendations:")
                for rec in recommendations:
                    typer.echo(f"  • {rec}")
        else:
            typer.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")


# =============================================================================
# Category: optimize
# =============================================================================

if typer and optimize_app is not None:

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
        from cli.commands import optimization
        _run(optimization.recommend, ctx, model_size=model_size, gpus=gpus, goal=goal)

    @optimize_app.command("auto", help="Auto-optimization with LLM guidance")
    def optimize_auto(ctx: typer.Context) -> None:
        from cli.commands import optimization
        _run(optimization.auto_optimize, ctx)

    @optimize_app.command("whatif", help="What-if analysis for optimizations")
    def optimize_whatif(ctx: typer.Context) -> None:
        from cli.commands import optimization
        _run(optimization.whatif, ctx)

    @optimize_app.command("stacking", help="Compound optimization stacking")
    def optimize_stacking(ctx: typer.Context) -> None:
        from cli.commands import optimization
        _run(optimization.stacking, ctx)

    @optimize_app.command("playbook", help="Pre-built optimization playbooks")
    def optimize_playbook(
        ctx: typer.Context,
        action: str = typer.Argument("list", help='Playbook name or "list"'),
    ) -> None:
        from cli.commands import optimization
        _run(optimization.playbook, ctx, action=action)


# =============================================================================
# Category: distributed
# =============================================================================

if typer and distributed_app is not None:

    @distributed_app.command("plan", help="Plan parallelism strategy (TP/PP/DP)")
    def distributed_plan(
        ctx: typer.Context,
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(8, "--gpus", help="Number of GPUs"),
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
    ) -> None:
        from cli.commands import distributed
        _run(distributed.plan_parallelism, ctx, model_size=model_size, gpus=gpus, nodes=nodes)

    @distributed_app.command("topology", help="Analyze GPU topology")
    def distributed_topology(ctx: typer.Context) -> None:
        from cli.commands import distributed
        _run(distributed.topology, ctx)

    @distributed_app.command("nccl", help="NCCL tuning recommendations")
    def distributed_nccl(
        ctx: typer.Context,
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
        gpus: int = typer.Option(8, "--gpus", help="GPUs per node"),
        diagnose: bool = typer.Option(False, "--diagnose", help="Include diagnostic checks"),
    ) -> None:
        from cli.commands import distributed
        _run(distributed.nccl_tuning, ctx, nodes=nodes, gpus=gpus, diagnose=diagnose)

    @distributed_app.command("zero", help="ZeRO/FSDP configuration")
    def distributed_zero(
        ctx: typer.Context,
        model_size: float = typer.Option(70.0, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(8, "--gpus", help="Number of GPUs"),
    ) -> None:
        from cli.commands import distributed
        _run(distributed.zero_config, ctx, model_size=model_size, gpus=gpus)

    @distributed_app.command("launch-plan", help="Generate torchrun launch plan (dry-run/save).")
    def distributed_launch_plan(
        ctx: typer.Context,
        model_params: int = typer.Option(70, "--model-params", help="Model size in billions"),
        nodes: int = typer.Option(1, "--nodes", help="Nodes"),
        gpus: int = typer.Option(8, "--gpus", help="GPUs per node"),
        tp: int = typer.Option(1, "--tp", help="Tensor parallel degree"),
        pp: int = typer.Option(1, "--pp", help="Pipeline parallel degree"),
        dp: int = typer.Option(1, "--dp", help="Data parallel degree"),
        batch_size: int = typer.Option(1, "--batch-size", help="Global batch size"),
        script: str = typer.Option("train.py", "--script", help="Entry script to launch"),
        extra_args: Optional[str] = typer.Option(None, "--extra-args", help="Extra args to append to command"),
        save_plan: Optional[Path] = typer.Option(None, "--save-plan", help="Optional path to save plan JSON"),
    ) -> None:
        from core.optimization.parallelism_planner.launch_plan import generate_launch_plan

        try:
            plan = generate_launch_plan(
                model_params=model_params,
                nodes=nodes,
                gpus_per_node=gpus,
                tp=tp,
                pp=pp,
                dp=dp,
                batch_size=batch_size,
                script=script,
                extra_args=extra_args,
            )
        except Exception as exc:
            typer.echo(f"Failed to build launch plan: {exc}", err=True)
            raise typer.Exit(code=1)

        typer.echo("\n🧭 Launch Plan")
        typer.echo(f"Model params: {plan.model_params}B")
        typer.echo(f"Layout: TP={plan.tp} PP={plan.pp} DP={plan.dp} on {plan.nodes}x{plan.gpus_per_node}")
        typer.echo(f"Command:\n  {plan.command}")
        if save_plan:
            save_plan.write_text(plan.to_json())
            typer.echo(f"\n💾 Saved plan to {save_plan}")


# =============================================================================
# Category: inference
# =============================================================================

if typer and inference_app is not None:

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
        from cli.commands import inference
        _run(inference.vllm_config, ctx, model=model, target=target)

    @inference_app.command("quantize", help="Quantization recommendations")
    def inference_quantize(
        ctx: typer.Context,
        model_size: float = typer.Option(70.0, "--model-size", help="Model size in billions"),
        target_memory: Optional[float] = typer.Option(
            None, "--target-memory", help="Target memory per GPU (GB)"
        ),
    ) -> None:
        from cli.commands import inference
        _run(inference.quantize, ctx, model_size=model_size, target_memory=target_memory)

    @inference_app.command("deploy", help="Deployment configuration")
    def inference_deploy(
        ctx: typer.Context,
        model: str = typer.Option("meta-llama/Llama-2-70b-hf", "--model", help="Model name"),
        target: str = typer.Option("vllm", "--target", help="Backend target (vllm/tensorrt/triton)"),
    ) -> None:
        from cli.commands import inference
        _run(inference.deploy_config, ctx, model=model, target=target)

    @inference_app.command("serve", help="Start inference server")
    def inference_serve(
        ctx: typer.Context,
        model: str = typer.Option("meta-llama/Llama-2-70b-hf", "--model", help="Model name"),
        gpus: int = typer.Option(1, "--gpus", help="Tensor parallel size"),
    ) -> None:
        from cli.commands import inference
        _run(inference.serve, ctx, model=model, gpus=gpus)


# =============================================================================
# Category: training
# =============================================================================

if typer and training_app is not None:

    @training_app.command("rl", help="RL/RLHF optimization")
    def training_rl(ctx: typer.Context) -> None:
        from cli.commands import training
        _run(training.rl, ctx)

    @training_app.command("checkpoint", help="Checkpointing strategy")
    def training_checkpoint(ctx: typer.Context) -> None:
        from cli.commands import training
        _run(training.checkpointing, ctx)

    @training_app.command("gradient", help="Gradient optimization")
    def training_gradient(ctx: typer.Context) -> None:
        from cli.commands import training
        _run(training.gradient, ctx)


# =============================================================================
# Category: monitor
# =============================================================================

if typer and monitoring_app is not None:

    @monitoring_app.command("live", help="Real-time GPU monitoring")
    def monitor_live(ctx: typer.Context) -> None:
        from cli.commands import monitoring
        _run(monitoring.live_monitor, ctx)

    @monitoring_app.command("regression", help="Detect performance regressions")
    def monitor_regression(ctx: typer.Context) -> None:
        from cli.commands import monitoring
        _run(monitoring.regression, ctx)

    @monitoring_app.command("metrics", help="Collect and display metrics")
    def monitor_metrics(ctx: typer.Context) -> None:
        from cli.commands import monitoring
        _run(monitoring.metrics, ctx)


# =============================================================================
# Category: report
# =============================================================================

if typer and report_app is not None:

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
        from cli.commands import reports
        _run(reports.generate_report, ctx, output=str(output) if output else None, format=fmt)

    @report_app.command("history", help="View optimization history")
    def report_history(ctx: typer.Context) -> None:
        from cli.commands import reports
        _run(reports.show_history, ctx)

    @report_app.command("roi", help="Calculate ROI of optimizations")
    def report_roi(ctx: typer.Context) -> None:
        from cli.commands import reports
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
        from cli.commands import reports
        _run(reports.export, ctx, format=fmt, output=str(output) if output else None)


# =============================================================================
# Category: hf (HuggingFace)
# =============================================================================

if typer and hf_app is not None:

    @hf_app.command("search", help="Search HuggingFace models")
    def hf_search(
        ctx: typer.Context,
        query: Optional[str] = typer.Argument(None, help="Search query"),
        task: Optional[str] = typer.Option(None, "--task", help="Filter by task"),
        limit: int = typer.Option(10, "--limit", help="Max results"),
    ) -> None:
        from cli.commands import huggingface
        _run(huggingface.search, ctx, query=query, task=task, limit=limit)

    @hf_app.command("trending", help="Trending models")
    def hf_trending(
        ctx: typer.Context,
        task: str = typer.Option("text-generation", "--task", help="Task type"),
        limit: int = typer.Option(15, "--limit", help="Max results"),
    ) -> None:
        from cli.commands import huggingface
        _run(huggingface.trending, ctx, task=task, limit=limit)

    @hf_app.command("model", help="Model information")
    def hf_model(
        ctx: typer.Context,
        model: Optional[str] = typer.Argument(None, help="Model ID"),
    ) -> None:
        from cli.commands import huggingface
        _run(huggingface.model_info, ctx, model=model)

    @hf_app.command("config", help="Generate optimal config")
    def hf_config(
        ctx: typer.Context,
        model: Optional[str] = typer.Argument(None, help="Model ID"),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
    ) -> None:
        from cli.commands import huggingface
        _run(huggingface.model_config, ctx, model=model, gpus=gpus)

    @hf_app.command("download", help="Download a HuggingFace model")
    def hf_download(
        ctx: typer.Context,
        model: str = typer.Argument(..., help="Model ID (e.g., 'meta-llama/Llama-3.1-8B')"),
        revision: Optional[str] = typer.Option(None, "--revision", "-r", help="Branch, tag, or commit"),
        cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Custom cache directory"),
        token: Optional[str] = typer.Option(None, "--token", help="HuggingFace token for gated models"),
        include: Optional[List[str]] = typer.Option(None, "--include", help="Patterns to include (e.g., '*.safetensors')"),
        exclude: Optional[List[str]] = typer.Option(None, "--exclude", help="Patterns to exclude"),
    ) -> None:
        """Download a model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            typer.echo("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
            raise typer.Exit(code=1)
        
        typer.echo(f"📥 Downloading: {model}")
        if revision:
            typer.echo(f"   Revision: {revision}")
        
        try:
            path = snapshot_download(
                model,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                token=token or os.environ.get("HF_TOKEN"),
                allow_patterns=include,
                ignore_patterns=exclude,
            )
            typer.echo(f"✅ Downloaded to: {path}")
        except Exception as e:
            typer.echo(f"❌ Download failed: {e}", err=True)
            raise typer.Exit(code=1)


# =============================================================================
# Category: hw (hardware benchmarks - was hwbench + test)
# =============================================================================

if typer:

    @hw_app.command("memory", help="GPU VRAM bandwidth test")
    def hw_memory(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Buffer size (MB)"),
        stride: int = typer.Option(128, "--stride", help="Stride (bytes)"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.mem_hierarchy, ctx, size_mb=size_mb, stride=stride)

    @hw_app.command("cache", help="Memory hierarchy stride test (cache behavior)")
    def hw_cache(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Buffer size (MB)"),
        stride: int = typer.Option(128, "--stride", help="Stride (bytes)"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.mem_hierarchy, ctx, size_mb=size_mb, stride=stride)

    @hw_app.command("roofline", help="Stride sweep ASCII roofline for memory")
    def hw_roofline(
        ctx: typer.Context,
        size_mb: int = typer.Option(32, "--size-mb", help="Buffer size (MB)"),
        strides: Optional[List[int]] = typer.Option(None, "--stride", help="Stride values (repeatable)"),
    ) -> None:
        from cli.commands import tests as test_cmds
        _run(test_cmds.mem_roofline, ctx, size_mb=size_mb, strides=strides)

    @hw_app.command("pcie", help="PCIe H2D/D2H bandwidth")
    def hw_pcie(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        iters: int = typer.Option(10, "--iters", "-i", help="Iterations"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.pcie, ctx, size_mb=size_mb, iters=iters)

    @hw_app.command("tc", help="Tensor core throughput (TFLOPS)")
    def hw_tc(
        ctx: typer.Context,
        size: int = typer.Option(4096, "--size", help="Matrix size"),
        precision: str = typer.Option("fp16", "--precision", help="Precision (fp16/fp32/bf16/fp8)"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.tensor_core, ctx, size=size, precision=precision)

    @hw_app.command("sfu", help="SFU (Special Function Units) benchmark")
    def hw_sfu(
        ctx: typer.Context,
        elements: int = typer.Option(64 * 1024 * 1024, "--elements", help="Number of elements"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.sfu, ctx, elements=elements)

    @hw_app.command("disk", help="Disk I/O benchmark (sequential)")
    def hw_disk(
        ctx: typer.Context,
        file_size_mb: int = typer.Option(256, "--size-mb", "-s", help="File size (MB)"),
        block_size_kb: int = typer.Option(1024, "--block-kb", "-b", help="Block size (KB)"),
        tmp_dir: Optional[Path] = typer.Option(None, "--tmp-dir", help="Temporary directory"),
    ) -> None:
        from core.diagnostics import microbench
        _run(
            microbench.disk,
            ctx,
            file_size_mb=file_size_mb,
            block_size_kb=block_size_kb,
            tmp_dir=str(tmp_dir) if tmp_dir else None,
        )

    @hw_app.command("tcp", help="TCP loopback throughput")
    def hw_tcp(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        port: int = typer.Option(5789, "--port", help="Port to use"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.loopback, ctx, size_mb=size_mb, port=port)

    @hw_app.command("ib", help="InfiniBand bandwidth test")
    def hw_ib(
        ctx: typer.Context,
        size_mb: int = typer.Option(64, "--size-mb", "-s", help="Transfer size (MB)"),
        duration: int = typer.Option(5, "--duration", "-d", help="Test duration (seconds)"),
    ) -> None:
        """Run InfiniBand bandwidth test using ib_write_bw or NCCL."""
        import shutil
        
        typer.echo("\n📡 InfiniBand Bandwidth Test\n")
        
        ib_write_bw = shutil.which("ib_write_bw")
        if ib_write_bw:
            typer.echo("Using: ib_write_bw (perftest)")
            typer.echo(f"Size: {size_mb} MB, Duration: {duration}s")
            typer.echo("\nTo run manually between two nodes:")
            typer.echo("  Server: ib_write_bw -d mlx5_0")
            typer.echo("  Client: ib_write_bw -d mlx5_0 <server_ip>")
        else:
            typer.echo("ib_write_bw not found. Install perftest package:")
            typer.echo("  apt install perftest  # or yum install perftest")
            typer.echo("\nAlternative: Use NCCL tests")
            typer.echo("  all_reduce_perf -b 8M -e 256M -g 8")

    @hw_app.command("nccl", help="NCCL collective bandwidth (all_reduce, etc.)")
    def hw_nccl(
        ctx: typer.Context,
        collective: str = typer.Option("all_reduce", "--collective", "-c", help="Collective type"),
        min_bytes: str = typer.Option("8M", "--min", help="Min message size"),
        max_bytes: str = typer.Option("256M", "--max", help="Max message size"),
        gpus: int = typer.Option(8, "--gpus", "-g", help="Number of GPUs"),
    ) -> None:
        """Run NCCL collective bandwidth test."""
        import shutil
        
        typer.echo("\n🔄 NCCL Collective Bandwidth Test\n")
        
        # Map collective name to binary
        collective_bins = {
            "all_reduce": "all_reduce_perf",
            "all_gather": "all_gather_perf", 
            "reduce_scatter": "reduce_scatter_perf",
            "broadcast": "broadcast_perf",
            "reduce": "reduce_perf",
            "alltoall": "alltoall_perf",
        }
        
        bin_name = collective_bins.get(collective, f"{collective}_perf")
        bin_path = shutil.which(bin_name)
        
        if bin_path:
            cmd = [bin_path, "-b", min_bytes, "-e", max_bytes, "-g", str(gpus)]
            typer.echo(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            raise typer.Exit(code=result.returncode)
        else:
            typer.echo(f"{bin_name} not found. Install nccl-tests:")
            typer.echo("  git clone https://github.com/NVIDIA/nccl-tests")
            typer.echo("  cd nccl-tests && make MPI=1")
            typer.echo(f"\nManual command:")
            typer.echo(f"  {bin_name} -b {min_bytes} -e {max_bytes} -g {gpus}")

    @hw_app.command("p2p", help="GPU-to-GPU P2P/NVLink bandwidth")
    def hw_p2p(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        bidirectional: bool = typer.Option(False, "--bidir", "-b", help="Bidirectional test"),
    ) -> None:
        """Test GPU-to-GPU peer-to-peer bandwidth over NVLink/PCIe."""
        try:
            import torch
        except ImportError:
            typer.echo("❌ PyTorch not available")
            raise typer.Exit(code=1)
        
        if not torch.cuda.is_available():
            typer.echo("❌ CUDA not available")
            raise typer.Exit(code=1)
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            typer.echo("❌ P2P test requires at least 2 GPUs")
            raise typer.Exit(code=1)
        
        typer.echo(f"\n🔗 GPU P2P Bandwidth Test ({num_gpus} GPUs)\n")
        typer.echo(f"Transfer size: {size_mb} MB")
        typer.echo(f"Mode: {'Bidirectional' if bidirectional else 'Unidirectional'}\n")
        
        size_bytes = size_mb * 1024 * 1024
        
        # Test P2P between all pairs
        typer.echo("GPU Pair    │ P2P Enabled │ Bandwidth (GB/s)")
        typer.echo("────────────┼─────────────┼─────────────────")
        
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i == j:
                    continue
                
                can_access = torch.cuda.can_device_access_peer(i, j)
                
                # Allocate tensors
                with torch.cuda.device(i):
                    src = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{i}")
                with torch.cuda.device(j):
                    dst = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{j}")
                
                # Warmup
                dst.copy_(src)
                torch.cuda.synchronize()
                
                # Benchmark
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                iters = 10
                start.record()
                for _ in range(iters):
                    dst.copy_(src)
                end.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start.elapsed_time(end)
                bw_gbps = (size_bytes * iters / (elapsed_ms / 1000)) / 1e9
                
                p2p_str = "✓" if can_access else "✗"
                typer.echo(f"GPU {i}→{j}    │ {p2p_str:^11} │ {bw_gbps:>8.2f}")
        
        typer.echo()

    @hw_app.command("speed", help="Run speed tests (GEMM, memory, attention)")
    def hw_speed(
        ctx: typer.Context,
        type: SpeedTestChoice = typer.Option(
            SpeedTestChoice.all,
            "--type",
            help="Speed test type",
            show_choices=True,
        ),
        gemm_size: int = typer.Option(512, "--gemm-size", help="Square GEMM size (e.g., 512 => 512x512)"),
        precision: str = typer.Option("fp16", "--precision", help="Precision for GEMM (fp16/bf16/tf32/fp32/fp8)"),
        mem_size_mb: int = typer.Option(16, "--mem-size-mb", help="Memory test size (MB)"),
        mem_stride: int = typer.Option(128, "--mem-stride", help="Memory test stride (bytes)"),
    ) -> None:
        from cli.commands import tests as test_cmds
        _run(
            test_cmds.speedtest,
            ctx,
            type=type,
            gemm_size=gemm_size,
            precision=precision,
            mem_size_mb=mem_size_mb,
            mem_stride=mem_stride,
        )

    @hw_app.command("diagnostics", help="System diagnostics")
    def hw_diagnostics(ctx: typer.Context) -> None:
        from cli.commands import tests as test_cmds
        _run(test_cmds.diagnostics, ctx)


# =============================================================================
# Category: cluster
# =============================================================================

if typer and cluster_app is not None:

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
        from cli.commands import cluster
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
        from cli.commands import cluster
        _run(cluster.cloud_cost, ctx, model_size=model_size, tokens=tokens, batch_size=batch_size)

    @cluster_app.command("scaling", help="Predict scaling efficiency")
    def cluster_scaling(
        ctx: typer.Context,
        gpus: int = typer.Option(8, "--gpus", help="Target GPUs"),
        model_size: float = typer.Option(70.0, "--model-size", help="Model size (B)"),
        batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    ) -> None:
        from cli.commands import cluster
        _run(cluster.scaling_predict, ctx, gpus=gpus, model_size=model_size, batch_size=batch_size)

    @cluster_app.command("diagnose", help="Cluster diagnostics")
    def cluster_diagnose(ctx: typer.Context) -> None:
        from cli.commands import cluster
        _run(cluster.cluster_diagnose, ctx)

    @cluster_app.command("power", help="Power analysis")
    def cluster_power(ctx: typer.Context) -> None:
        from cli.commands import cluster
        _run(cluster.power_analysis, ctx)


# =============================================================================
# Monitoring runners (extended)
# =============================================================================

if typer:

    @monitoring_app.command("cluster", help="Cluster monitor (agent/master) runner.")
    def monitoring_cluster(
        args: List[str] = typer.Argument(None, help="Arguments forwarded to monitoring.cluster_monitor"),
    ) -> None:
        _run_module("monitoring.cluster_monitor", args or [])

    @monitoring_app.command("prometheus", help="Start Prometheus exporter for GPU metrics.")
    def monitoring_prometheus(
        args: List[str] = typer.Argument(None, help="Arguments forwarded to monitoring.prometheus_exporter"),
    ) -> None:
        _run_module("monitoring.prometheus_exporter", args or [])


# =============================================================================
# Top-level commands (bench, version, help)
# =============================================================================

if typer:

    # Bench integration (core)
    try:
        from core.benchmark import bench_commands
        BENCH_APP = bench_commands.app if getattr(bench_commands, "TYPER_AVAILABLE", False) else None
    except Exception:
        BENCH_APP = None

    plugin_apps = load_plugin_apps()

    if BENCH_APP:
        app.add_typer(BENCH_APP, name="bench", help="Run and profile benchmarks")
    else:  # pragma: no cover - bench missing during docs builds

        @app.command("bench", help="Run and profile benchmarks")
        def bench_stub() -> None:
            typer.echo("Bench CLI unavailable (typer not installed or import failed).")
            raise typer.Exit(code=1)

    # Top-level utility commands
    @app.command("tui", help="Launch interactive terminal UI")
    def tui_command(
        ctx: typer.Context,
        data_file: Optional[Path] = typer.Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ) -> None:
        from cli.tui import run_tui
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
        from dashboard.api.server import serve_dashboard
        serve_dashboard(port=port, data_file=data, open_browser=not no_browser)

    @app.command("mcp", help="Start MCP server for AI chat integration")
    def mcp_command(
        ctx: typer.Context,
        list_tools: bool = typer.Option(False, "--list", help="List available tools"),
        test: Optional[str] = typer.Option(None, "--test", help="Test a specific tool"),
        serve: bool = typer.Option(False, "--serve", help="Start MCP server (stdio)"),
    ) -> None:
        from mcp import server

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
# Attach sub-apps (10 domains + legacy aliases)
# =============================================================================

if typer:
    # Primary domain apps (the 10-domain model)
    app.add_typer(gpu_app, name="gpu", help="GPU: info, topology, power, bandwidth")
    app.add_typer(system_app, name="system", help="System: software, deps, capabilities")
    app.add_typer(profile_app, name="profile", help="Profile: nsys, ncu, flame, HTA")
    app.add_typer(analyze_app, name="analyze", help="Analyze: bottlenecks, pareto, scaling")
    app.add_typer(optimize_app, name="optimize", help="Optimize: recommend, ROI, techniques")
    app.add_typer(distributed_app, name="distributed", help="Distributed: plan, NCCL, FSDP")
    app.add_typer(inference_app, name="inference", help="Inference: vLLM, quantization")
    app.add_typer(benchmark_app, name="benchmark", help="Benchmark: run, targets, history")
    app.add_typer(ai_app, name="ai", help="AI: ask, explain, suggest")
    app.add_typer(export_app, name="export", help="Export: CSV, PDF, HTML")
    
    # Legacy aliases (for backward compat, same underlying apps)
    app.add_typer(gpu_app, name="info", help="[Alias for 'gpu']", hidden=True)
    app.add_typer(benchmark_app, name="hw", help="[Alias for 'benchmark']", hidden=True)
    app.add_typer(export_app, name="report", help="[Alias for 'export']", hidden=True)
    
    # Legacy apps (will be phased out)
    app.add_typer(monitoring_app, name="monitor", hidden=True)
    app.add_typer(training_app, name="training", hidden=True)
    app.add_typer(hf_app, name="hf", hidden=True)
    app.add_typer(cluster_app, name="cluster", hidden=True)


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
