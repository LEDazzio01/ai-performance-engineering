"""
Analysis Commands - Profiling, comparison, roofline analysis.

Delegates to existing implementations in tools/analysis/ and tools/profiling/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _print_header(title: str, emoji: str = "🔬"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


def _run_bench_cli(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "cli.aisp", "bench", *args]
    proc = subprocess.run(cmd)
    return proc.returncode


def run_profile(args) -> int:
    """Run deep profiling via the benchmark harness."""
    _print_header("Profile Analysis", "📊")

    targets = getattr(args, "targets", None)
    profile = getattr(args, "profile", "deep_dive")
    if not targets:
        print("  Usage: aisp analyze profile --targets chX[:example] [--profile deep_dive]")
        return 1

    cli_args = ["run", "--profile", profile, "--llm-analysis"]
    for t in targets:
        cli_args.extend(["-t", t])

    print(f"  Running bench with profile='{profile}' on targets: {', '.join(targets)}")
    return _run_bench_cli(cli_args)


def compare_runs(args) -> int:
    """Compare baseline/optimized pairs using the existing comparator."""
    _print_header("Run Comparison", "⚖️")

    chapter = getattr(args, "chapter", None) or "labs/moe_parallelism"
    targets = getattr(args, "targets", None) or []
    cli_args = ["--chapter", chapter]
    if targets:
        cli_args.extend(["--targets", ",".join(targets)])

    try:
        script = Path("tools/analysis/compare_benchmark_pairs.py")
        cmd = [sys.executable, str(script), *cli_args]
        print(f"  Invoking comparator: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode
    except Exception as e:
        print(f"  Error running comparator: {e}")
        return 1


def diff_analysis(args) -> int:
    """Differential analysis between baseline and optimized deep profile JSON."""
    _print_header("Differential Analysis", "📈")

    baseline = getattr(args, "baseline", None)
    optimized = getattr(args, "optimized", None)
    if not baseline or not optimized:
        print("  Usage: aisp analyze diff <baseline.json> <optimized.json>")
        return 1

    try:
        from tools.analysis.differential_profile_analyzer import analyze_differential

        report = analyze_differential(Path(baseline), Path(optimized))
        data = report.to_dict()
        print(json.dumps(data, indent=2))
        return 0
    except FileNotFoundError as e:
        print(f"  File not found: {e}")
        return 1
    except Exception as e:
        print(f"  Error: {e}")
        return 1


def roofline(args) -> int:
    """Roofline model analysis."""
    _print_header("Roofline Analysis", "📐")

    try:
        from tools.analysis.roofline_automation import generate_roofline

        generate_roofline()
        print("  Roofline generation completed.")
        return 0
    except ImportError:
        print("  Roofline tool not available")
    except Exception as e:
        print(f"  Error: {e}")
    return 1


def bottleneck(args) -> int:
    """Identify performance bottlenecks."""
    _print_header("Bottleneck Analysis", "🔍")

    try:
        from tools.core.engine import get_engine

        engine = get_engine()
        mode = getattr(args, "mode", "both")
        limit = getattr(args, "limit", 5)
        raw = engine.analyze.bottlenecks("profile" if mode == "profile" else "bottleneck")

        if args.json:
            import json
            print(json.dumps(raw, indent=2, default=str))
            return 0

        profile = raw.get("profile", {}) if isinstance(raw, dict) else {}
        bottlenecks = profile.get("bottlenecks", []) if isinstance(profile, dict) else []
        llm = raw.get("llm") if isinstance(raw, dict) else None
        llm_text = llm.get("llm_response") if isinstance(llm, dict) else None

        if profile.get("error"):
            print(f"  Profile analysis unavailable: {profile['error']}")
        elif not bottlenecks:
            print("  No profile bottlenecks detected (or no traces found).")
        else:
            print("  Top bottlenecks from profile data:")
            for b in bottlenecks[: max(1, limit)]:
                pct = b.get("percentage")
                label = f"{b.get('type', 'unknown')}"
                if b.get("kernel_name"):
                    label += f" ({b['kernel_name']})"
                print(f"   - {label}: {pct}%  | {b.get('description', '')}")

        if mode in ("both", "llm") or (not bottlenecks and llm_text):
            if llm_text:
                print("\n  LLM guidance:")
                print(llm_text)
            elif llm and llm.get("error"):
                print(f"\n  LLM analysis unavailable: {llm['error']}")
            else:
                print("\n  No LLM guidance available. Use --mode llm to force LLM-only analysis.")

    except Exception as e:
        print(f"  Error running bottleneck analysis: {e}")
        return 1

    return 0
