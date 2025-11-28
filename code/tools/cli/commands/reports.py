"""
Reports Commands - Report generation, history, ROI calculation, exports.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tools.analysis.performance_analyzer import load_benchmark_data, PerformanceAnalyzer
from tools.core.perf_core import get_core


def _print_header(title: str, emoji: str = "📄"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def generate_report(args) -> int:
    """Generate performance report."""
    _print_header("Performance Report", "📊")
    
    fmt = getattr(args, "format", "markdown")
    output = getattr(args, "output", None) or (
        Path("artifacts/reports/report.md") if fmt == "markdown" else Path("artifacts/reports/report.json")
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = load_benchmark_data()
    analyzer = PerformanceAnalyzer(lambda: data)
    leaderboards = analyzer.get_categorized_leaderboards()
    summary = data.get("summary", {})

    if fmt == "json":
        payload = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "leaderboards": leaderboards,
        }
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  ✅ JSON report written to {output}")
    else:
        lines = [
            "# AI Performance Report",
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            f"- Total benchmarks: {summary.get('total_benchmarks', 0)}",
            f"- Avg speedup: {summary.get('avg_speedup', 0):.2f}x",
            f"- Max speedup: {summary.get('max_speedup', 0):.2f}x",
            f"- Successful: {summary.get('successful', 0)}",
            f"- Failed: {summary.get('failed', 0)}",
            "",
            "## Speed Leaders",
        ]
        speed = leaderboards.get("leaderboards", {}).get("speed", {}).get("entries", [])[:5]
        for entry in speed:
            lines.append(f"- {entry.get('name')} — {entry.get('primary_metric')}")

        lines.append("\n## Memory Leaders")
        memory = leaderboards.get("leaderboards", {}).get("memory", {}).get("entries", [])[:5]
        for entry in memory:
            lines.append(f"- {entry.get('name')} — {entry.get('primary_metric')}")

        output.write_text("\n".join(lines))
        print(f"  ✅ Markdown report written to {output}")
    
    return 0


def show_history(args) -> int:
    """View optimization history."""
    _print_header("Optimization History", "📜")
    
    data = load_benchmark_data()
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        print("  No benchmark history found.")
        return 1

    # Sort by timestamp if present else speedup
    benchmarks = sorted(
        benchmarks,
        key=lambda b: b.get("timestamp", "") or b.get("speedup", 0),
        reverse=True,
    )

    total_speedups = sum(b.get("speedup", 1.0) for b in benchmarks)
    print("\n  Recent benchmark results (top 10):")
    for b in benchmarks[:10]:
        print(f"    {b.get('chapter','?')}:{b.get('name','?')} → {b.get('speedup',1.0):.2f}x")

    avg_speedup = total_speedups / max(len(benchmarks), 1)
    print(f"\n  Average speedup across {len(benchmarks)} results: {avg_speedup:.2f}x")
    return 0


def calculate_roi(args) -> int:
    """Calculate ROI of optimizations."""
    _print_header("ROI Calculator", "💰")
    
    data = load_benchmark_data()
    summary = data.get("summary", {})
    avg_speedup = summary.get("avg_speedup", 1.0) or 1.0

    before_throughput = 1000.0
    after_throughput = before_throughput * avg_speedup
    hourly_cost = 3.00  # placeholder infra cost

    before_cost_per_1m = (hourly_cost / before_throughput * 1_000_000) / 3600
    after_cost_per_1m = (hourly_cost / after_throughput * 1_000_000) / 3600
    savings_pct = (1 - after_cost_per_1m / before_cost_per_1m) * 100
    
    print("\n  Cost Analysis (using average speedup from benchmarks):")
    print(f"    Avg speedup:        {avg_speedup:.2f}x")
    print(f"    Before optimization: ${before_cost_per_1m:.3f} per 1M tokens")
    print(f"    After optimization:  ${after_cost_per_1m:.3f} per 1M tokens")
    print(f"    Savings:             {savings_pct:.1f}%")
    
    print("\n  At 1B tokens/month:")
    monthly_before = before_cost_per_1m * 1000
    monthly_after = after_cost_per_1m * 1000
    monthly_savings = monthly_before - monthly_after
    
    print(f"    Before: ${monthly_before:,.0f}/month")
    print(f"    After:  ${monthly_after:,.0f}/month")
    print(f"    Saved:  ${monthly_savings:,.0f}/month")
    
    return 0


def export(args) -> int:
    """Export results to various formats."""
    _print_header("Export Results", "📤")

    fmt = getattr(args, "format", "json")
    output = getattr(args, "output", None)
    core = get_core()

    if not output:
        suffix = {
            "json": "json",
            "csv": "csv",
            "detailed_csv": "csv",
            "html": "html",
        }.get(fmt, "json")
        output = Path(f"artifacts/exports/benchmarks.{suffix}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "json":
            data = load_benchmark_data()
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
        elif fmt == "csv":
            output.write_text(core.export_benchmarks_csv())
        elif fmt == "detailed_csv":
            output.write_text(core.export_detailed_csv())
        elif fmt == "html":
            data = load_benchmark_data()
            html = _render_html_summary(data)
            output.write_text(html)
        else:
            print(f"  ❌ Unknown format: {fmt}")
            return 1

        print(f"  ✅ Exported {fmt} to {output}")
        return 0
    except Exception as e:
        print(f"  ❌ Error exporting: {e}")
        return 1


def _render_html_summary(data: dict) -> str:
    """Render a lightweight HTML summary of benchmarks."""
    benchmarks = data.get("benchmarks", [])
    rows = []
    for b in benchmarks[:50]:
        rows.append(
            f"<tr><td>{b.get('chapter','')}</td><td>{b.get('name','')}</td>"
            f"<td>{b.get('baseline_time_ms',0):.2f}</td>"
            f"<td>{b.get('optimized_time_ms',0):.2f}</td>"
            f"<td>{b.get('speedup',1):.2f}x</td></tr>"
        )
    return f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Benchmark Export</title>
<style>body{{font-family:Arial,sans-serif;background:#0b1220;color:#e6edf7;padding:20px}}
table{{border-collapse:collapse;width:100%;background:#0f1629}}th,td{{border:1px solid #22304f;padding:8px}}th{{background:#16213d}}</style>
</head><body>
<h1>Benchmark Export</h1>
<table>
<thead><tr><th>Chapter</th><th>Name</th><th>Baseline (ms)</th><th>Optimized (ms)</th><th>Speedup</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body></html>
"""
