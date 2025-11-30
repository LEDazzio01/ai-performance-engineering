"""CLI commands for profile analysis and comparison."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Optional, Any, Dict, List
import typer

from core.profile_insights import generate_flamegraph_comparison
from core.perf_core_base import PerformanceCoreBase


def _get_core() -> PerformanceCoreBase:
    """Get an instance of PerformanceCoreBase for profile operations."""
    return PerformanceCoreBase()


def compare_profiles(args) -> None:
    """Compare baseline vs optimized profiles and generate flame graph visualization."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    # Extract arguments from namespace
    chapter = getattr(args, 'chapter', None)
    output = getattr(args, 'output', 'comparison_flamegraph.html')
    json_out = getattr(args, 'json_out', None)
    
    if not chapter:
        # List available profile pairs
        core = _get_core()
        pairs = core.list_deep_profile_pairs()
        
        if not pairs.get("pairs"):
            console.print("[yellow]No profile pairs found.[/yellow]")
            console.print("Run profiling with nsys/ncu on baseline and optimized code first:")
            console.print("  [cyan]aisp profile nsys baseline_code.py[/cyan]")
            console.print("  [cyan]aisp profile nsys optimized_code.py[/cyan]")
            return
        
        table = Table(title="Available Profile Pairs for Comparison")
        table.add_column("Chapter", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("NSYS", style="yellow")
        table.add_column("NCU", style="yellow")
        
        for pair in pairs["pairs"]:
            table.add_row(
                pair.get("chapter", "unknown"),
                pair.get("type", "unknown"),
                "✓" if pair.get("has_nsys") else "✗",
                "✓" if pair.get("has_ncu") else "✗",
            )
        
        console.print(table)
        console.print(f"\nFound [green]{pairs['count']}[/green] profile pairs.")
        console.print("\nTo compare a specific chapter:")
        console.print("  [cyan]aisp profile compare <chapter-name>[/cyan]")
        return
    
    # Generate comparison for specified chapter
    core = _get_core()
    profile_dir = core._find_profile_directory(chapter)
    
    if not profile_dir:
        console.print(f"[red]Profile directory not found for: {chapter}[/red]")
        return
    
    console.print(f"[cyan]Analyzing profiles in: {profile_dir}[/cyan]")
    
    result = generate_flamegraph_comparison(profile_dir)
    
    if not result:
        console.print("[red]No baseline/optimized nsys profiles found.[/red]")
        return
    
    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        return
    
    # Display summary
    baseline = result.get("baseline", {})
    optimized = result.get("optimized", {})
    metrics = result.get("metrics", {})
    speedup = result.get("speedup", 1.0)
    
    console.print(Panel.fit(
        f"[bold green]⚡ {speedup}x Speedup[/bold green]\n\n"
        f"Baseline: {baseline.get('total_time_ms', 0):.2f} ms\n"
        f"Optimized: {optimized.get('total_time_ms', 0):.2f} ms\n\n"
        f"Sync Calls: {metrics.get('baseline_sync_calls', 0)} → {metrics.get('optimized_sync_calls', 0)} "
        f"([green]-{metrics.get('sync_reduction_pct', 0):.1f}%[/green])\n"
        f"Device Syncs: {metrics.get('baseline_device_sync', 0)} → {metrics.get('optimized_device_sync', 0)} "
        f"([green]-{metrics.get('device_sync_reduction_pct', 0):.1f}%[/green])\n"
        f"Wait Events: {metrics.get('optimized_wait_events', 0)} (lightweight coordination)",
        title="🔥 Flame Graph Comparison",
        border_style="green",
    ))
    
    if result.get("insight"):
        console.print(f"\n[yellow]Insight:[/yellow] {result['insight']}")
    
    # Generate HTML flame graph
    html_content = _generate_comparison_html(result, chapter)
    output_path = Path(output)
    output_path.write_text(html_content)
    console.print(f"\n[green]✓ Flame graph saved to: {output_path}[/green]")
    
    # Optionally save JSON
    if json_out:
        json_path = Path(json_out)
        json_path.write_text(json.dumps(result, indent=2))
        console.print(f"[green]✓ JSON data saved to: {json_path}[/green]")


def _generate_comparison_html(data: Dict[str, Any], chapter: str) -> str:
    """Generate an interactive HTML flame graph comparison."""
    baseline = data.get("baseline", {})
    optimized = data.get("optimized", {})
    metrics = data.get("metrics", {})
    speedup = data.get("speedup", 1.0)
    insight = data.get("insight", "")
    
    def bar_html(bars: list, is_kernel: bool = False) -> str:
        if not bars:
            return '<div class="bar other" style="width:100%">No data</div>'
        
        type_colors = {
            'sync': '#ef4444',
            'malloc': '#8b5cf6',
            'launch': '#a78bfa',
            'memcpy': '#fbbf24',
            'wait': '#34d399',
            'other': '#6b7280',
        }
        kernel_color = '#22d3d3'
        
        html_parts = []
        for bar in bars:
            color = kernel_color if is_kernel else type_colors.get(bar.get('type', 'other'), '#6b7280')
            width = max(bar.get('time_pct', 0), 3)
            name = bar.get('name', 'unknown')[:25]
            pct = bar.get('time_pct', 0)
            html_parts.append(
                f'<div class="bar" style="width:{width}%;background:{color}" '
                f'title="{bar.get("name", "")}: {pct:.1f}%">{name if pct > 8 else f"{pct:.0f}%"}</div>'
            )
        return ''.join(html_parts)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flame Graph Comparison - {chapter}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            font-size: 2rem; 
            margin-bottom: 1.5rem;
            background: linear-gradient(90deg, #22d3ee, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .speedup-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            border-radius: 9999px;
            font-size: 1.25rem;
            font-weight: bold;
            color: #0f172a;
            margin-bottom: 2rem;
        }}
        .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
        .panel {{
            background: rgba(255,255,255,0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .panel.baseline {{ border-left: 4px solid #ef4444; }}
        .panel.optimized {{ border-left: 4px solid #10b981; }}
        .panel h2 {{ 
            font-size: 1.1rem; 
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .panel h2 .time {{
            font-family: monospace;
            font-size: 0.9rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
        }}
        .panel.baseline h2 .time {{ background: rgba(239,68,68,0.2); color: #fca5a5; }}
        .panel.optimized h2 .time {{ background: rgba(16,185,129,0.2); color: #6ee7b7; }}
        .section-title {{ 
            font-size: 0.75rem; 
            text-transform: uppercase; 
            letter-spacing: 0.1em;
            color: rgba(255,255,255,0.5);
            margin: 1rem 0 0.5rem;
        }}
        .flame-row {{
            display: flex;
            gap: 2px;
            height: 2rem;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 0.5rem;
            font-size: 0.7rem;
            font-weight: 500;
            color: white;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: filter 0.2s;
            cursor: pointer;
        }}
        .bar:hover {{ filter: brightness(1.2); }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric {{
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .metric-label {{ font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-bottom: 0.5rem; }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; }}
        .metric-change {{ font-size: 0.8rem; color: #10b981; }}
        .insight {{
            background: linear-gradient(90deg, rgba(16,185,129,0.1), rgba(6,182,212,0.1));
            border: 1px solid rgba(16,185,129,0.3);
            border-radius: 0.75rem;
            padding: 1rem 1.5rem;
        }}
        .insight-title {{ color: #6ee7b7; font-weight: 600; margin-bottom: 0.5rem; }}
        .legend {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
            font-size: 0.8rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .legend-color {{
            width: 1rem;
            height: 0.75rem;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Flame Graph Comparison: {chapter}</h1>
        
        <div class="speedup-badge">⚡ {speedup}x Speedup</div>
        
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background:#ef4444"></div>Sync</div>
            <div class="legend-item"><div class="legend-color" style="background:#8b5cf6"></div>Malloc/Free</div>
            <div class="legend-item"><div class="legend-color" style="background:#a78bfa"></div>Launch</div>
            <div class="legend-item"><div class="legend-color" style="background:#fbbf24"></div>Memcpy</div>
            <div class="legend-item"><div class="legend-color" style="background:#34d399"></div>Wait</div>
            <div class="legend-item"><div class="legend-color" style="background:#22d3d3"></div>Kernel</div>
        </div>
        
        <div class="panels">
            <div class="panel baseline">
                <h2>
                    Baseline (Sequential)
                    <span class="time">{baseline.get('total_time_ms', 0):.2f} ms</span>
                </h2>
                <div class="section-title">CUDA API Distribution</div>
                <div class="flame-row">{bar_html(baseline.get('api_bars', []))}</div>
                <div class="section-title">Kernel Distribution</div>
                <div class="flame-row">{bar_html(baseline.get('kernel_bars', []), True)}</div>
            </div>
            <div class="panel optimized">
                <h2>
                    Optimized (Pipelined)
                    <span class="time">{optimized.get('total_time_ms', 0):.2f} ms</span>
                </h2>
                <div class="section-title">CUDA API Distribution</div>
                <div class="flame-row">{bar_html(optimized.get('api_bars', []))}</div>
                <div class="section-title">Kernel Distribution</div>
                <div class="flame-row">{bar_html(optimized.get('kernel_bars', []), True)}</div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Sync Calls</div>
                <div class="metric-value">{metrics.get('baseline_sync_calls', 0)} → {metrics.get('optimized_sync_calls', 0)}</div>
                <div class="metric-change">-{metrics.get('sync_reduction_pct', 0):.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Device Syncs</div>
                <div class="metric-value">{metrics.get('baseline_device_sync', 0)} → {metrics.get('optimized_device_sync', 0)}</div>
                <div class="metric-change">-{metrics.get('device_sync_reduction_pct', 0):.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Wait Events</div>
                <div class="metric-value">{metrics.get('optimized_wait_events', 0)}</div>
                <div class="metric-change">Lightweight sync</div>
            </div>
            <div class="metric">
                <div class="metric-label">Time Saved</div>
                <div class="metric-value">{(baseline.get('total_time_ms', 0) - optimized.get('total_time_ms', 0)):.1f} ms</div>
                <div class="metric-change">{((1 - optimized.get('total_time_ms', 1) / max(baseline.get('total_time_ms', 1), 0.001)) * 100):.1f}% faster</div>
            </div>
        </div>
        
        {"" if not insight else f'''<div class="insight">
            <div class="insight-title">💡 Optimization Insight</div>
            <p>{insight}</p>
        </div>'''}
    </div>
</body>
</html>'''


# Placeholder implementations for other profile commands
def flame(args) -> None:
    """Generate flame graph from profile."""
    from rich.console import Console
    file = getattr(args, 'file', None)
    console = Console()
    console.print(f"[yellow]Generating flame graph from {file or 'default profile'}...[/yellow]")
    console.print(f"[cyan]Use 'aisp profile compare' for baseline vs optimized comparison.[/cyan]")


def memory(args) -> None:
    """Memory timeline analysis."""
    from rich.console import Console
    file = getattr(args, 'file', None)
    console = Console()
    console.print(f"[yellow]Memory analysis for {file or 'default profile'}[/yellow]")


def kernels(args) -> None:
    """Kernel breakdown analysis."""
    from rich.console import Console
    file = getattr(args, 'file', None)
    console = Console()
    console.print(f"[yellow]Kernel breakdown for {file or 'default profile'}[/yellow]")


def hta(args) -> None:
    """Holistic Trace Analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Use 'aisp profile hta-capture' to collect a new HTA trace or view highlights in the dashboard.[/yellow]")


def torch_profiler(args) -> int:
    """Run torch.profiler capture with NVTX + lineinfo defaults."""
    from rich.console import Console
    from rich.panel import Panel
    from core.profiling.torch_profiler import TorchProfilerAutomation

    console = Console()
    script_path = Path(getattr(args, "script", ""))
    if not script_path.exists():
        console.print(f"[red]Script not found:[/red] {script_path}")
        return 1

    output_dir_opt = getattr(args, "output_dir", None)
    output_root = Path(output_dir_opt) if output_dir_opt else Path("artifacts/torch-profiles")
    mode = getattr(args, "mode", "full")
    output_name = getattr(args, "output_name", None) or script_path.stem
    nvtx_label = getattr(args, "nvtx_label", "aisp_torch_profile")
    force_lineinfo = bool(getattr(args, "force_lineinfo", True))
    use_nvtx = bool(getattr(args, "use_nvtx", True))
    timeout_seconds = getattr(args, "timeout_seconds", None)
    script_args: List[str] = getattr(args, "script_args", None) or []

    profiler = TorchProfilerAutomation(output_root)
    console.print(f"[cyan]Running torch.profiler ({mode}) on {script_path}[/cyan]")
    result = profiler.profile(
        script=script_path,
        output_name=output_name,
        mode=mode,
        script_args=script_args,
        force_lineinfo=force_lineinfo,
        timeout_seconds=timeout_seconds,
        nvtx_label=nvtx_label,
        use_nvtx=use_nvtx,
    )

    if not result.get("success"):
        console.print(f"[red]Profiler failed:[/red] {result.get('error', 'unknown error')}")
        return 1

    panel_lines = [
        f"[green]✓[/green] Trace: {result.get('trace_path') or 'n/a'}",
        f"Summary: {result.get('summary') and 'torch_profile_summary.json' or 'n/a'}",
        f"Metadata: {result.get('metadata') and 'metadata.json' or 'n/a'}",
        f"NVTX label: {result.get('nvtx_label')}",
        f"Force lineinfo: {result.get('force_lineinfo')}",
    ]
    console.print(Panel.fit("\n".join(panel_lines), title="torch.profiler capture", border_style="green"))
    return 0


def hta_capture(args) -> int:
    """Run an HTA-friendly nsys capture and analyze it."""
    from rich.console import Console
    from rich.panel import Panel
    from core.profiling.hta_capture import HTACaptureAutomation

    console = Console()
    command_str = getattr(args, "command", "")
    command_list = getattr(args, "command_list", None) or (shlex.split(command_str) if command_str else [])
    if not command_list:
        console.print("[red]Command is required (e.g., python train.py --arg foo)[/red]")
        return 1

    output_dir_opt = getattr(args, "output_dir", None)
    output_root = Path(output_dir_opt) if output_dir_opt else Path("artifacts/hta")
    output_name = getattr(args, "output_name", "hta_capture")
    preset = getattr(args, "preset", "full")
    force_lineinfo = bool(getattr(args, "force_lineinfo", True))
    timeout_seconds = getattr(args, "timeout_seconds", None)

    capture = HTACaptureAutomation(output_root)
    console.print(f"[cyan]Running HTA capture via nsys ({preset})[/cyan]")
    result = capture.capture(
        command=command_list,
        output_name=output_name,
        preset=preset,
        force_lineinfo=force_lineinfo,
        timeout_seconds=timeout_seconds,
    )

    if not result.get("success"):
        console.print(f"[red]HTA capture failed:[/red] {result.get('error', 'unknown error')}")
        return 1

    lines = [
        f"[green]✓[/green] NSYS report: {result.get('nsys_output')}",
        f"Trace JSON: {result.get('trace_path')}",
        f"HTA report: {result.get('hta_report')}",
        f"Force lineinfo: {result.get('force_lineinfo')}",
    ]
    console.print(Panel.fit("\n".join(lines), title="HTA capture", border_style="green"))
    return 0


def ncu(args) -> None:
    """NCU deep dive analysis."""
    from rich.console import Console
    script = getattr(args, 'script', None)
    console = Console()
    console.print(f"[yellow]NCU analysis for {script or 'default script'}[/yellow]")


def warp_divergence(args) -> None:
    """Warp divergence analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Warp divergence analysis[/yellow]")


def bank_conflicts(args) -> None:
    """Bank conflict analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Bank conflict analysis[/yellow]")


def occupancy(args) -> None:
    """Occupancy analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Occupancy analysis[/yellow]")
