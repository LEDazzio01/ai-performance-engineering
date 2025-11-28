"""
Shared helpers for exporting benchmark data and loading profiling artifacts.

These utilities are reused by the dashboard handler, CLI, and MCP layers to
avoid duplicating file-system scans and CSV/report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

CODE_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# Benchmark exports
# =============================================================================

def export_benchmarks_csv(data: Dict[str, Any]) -> str:
    """Export benchmark results to CSV format."""
    benchmarks = data.get("benchmarks", [])

    lines = [
        "Benchmark,Baseline Time (ms),Optimized Time (ms),Speedup,Chapter/Lab,Status"
    ]

    for b in benchmarks:
        name = b.get("name", "Unknown")
        baseline = b.get("baseline_time_ms", 0)
        optimized = b.get("optimized_time_ms", 0)
        speedup = b.get("speedup", 1.0)
        chapter = b.get("chapter", "")
        status = "✓ Optimized" if speedup > 1.1 else "⚠ Needs Work"

        lines.append(
            f'"{name}",{baseline:.4f},{optimized:.4f},{speedup:.2f},"{chapter}","{status}"'
        )

    return "\n".join(lines)


def export_detailed_csv(data: Dict[str, Any]) -> str:
    """Export detailed benchmark results including all metrics."""
    benchmarks = data.get("benchmarks", [])

    lines = [
        "Benchmark,Chapter,Baseline Time (ms),Optimized Time (ms),Speedup,"
        "Baseline Memory (MB),Optimized Memory (MB),Memory Reduction (%),"
        "Techniques Applied,LLM Patches,Patch Success Rate"
    ]

    for b in benchmarks:
        name = b.get("name", "Unknown")
        chapter = b.get("chapter", "")
        baseline = b.get("baseline_time_ms", 0)
        optimized = b.get("optimized_time_ms", 0)
        speedup = b.get("speedup", 1.0)
        baseline_mem = b.get("baseline_memory_mb", 0)
        optimized_mem = b.get("optimized_memory_mb", 0)
        mem_reduction = (
            (baseline_mem - optimized_mem) / baseline_mem * 100 if baseline_mem > 0 else 0
        )
        techniques = b.get("techniques", [])
        techniques_str = "; ".join(techniques) if techniques else ""
        llm_patches = b.get("llm_patches_applied", 0)
        patch_success = b.get("patch_success_rate", 0)

        lines.append(
            f'"{name}","{chapter}",{baseline:.4f},{optimized:.4f},{speedup:.2f},'
            f"{baseline_mem:.2f},{optimized_mem:.2f},{mem_reduction:.1f},"
            f'"{techniques_str}",{llm_patches},{patch_success:.0f}'
        )

    return "\n".join(lines)


# =============================================================================
# Profiling artifact loaders
# =============================================================================

def load_flame_graph_data(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load flame graph data from the most recent Chrome trace."""
    flame_data: Dict[str, Any] = {
        "name": "GPU Execution",
        "value": 0,
        "children": [],
    }

    profile_dirs = [
        code_root / "benchmark_profiles",
        code_root / "artifacts" / "profiles",
        code_root / "profiles",
    ]

    trace_files: List[Path] = []
    for profile_dir in profile_dirs:
        if profile_dir.exists():
            trace_files.extend(profile_dir.glob("**/*.json"))

    if trace_files:
        trace_files = sorted(trace_files, key=lambda f: f.stat().st_mtime, reverse=True)
        try:
            with open(trace_files[0]) as f:
                trace = json.load(f)

            events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

            kernel_times: Dict[str, Dict[str, float]] = {}
            for event in events:
                if event.get("ph") == "X" and event.get("dur", 0) > 10:
                    name = event.get("name", "unknown")
                    cat = event.get("cat", "other")
                    dur = event.get("dur", 0)

                    # Group by category
                    if cat not in kernel_times:
                        kernel_times[cat] = {}
                    if name not in kernel_times[cat]:
                        kernel_times[cat][name] = 0
                    kernel_times[cat][name] += dur

            for cat, kernels in kernel_times.items():
                cat_node = {
                    "name": cat,
                    "value": sum(kernels.values()),
                    "children": [
                        {"name": k, "value": v, "children": []}
                        for k, v in sorted(kernels.items(), key=lambda x: -x[1])[:20]
                    ],
                }
                flame_data["children"].append(cat_node)

            flame_data["value"] = sum(c["value"] for c in flame_data["children"])
            flame_data["trace_file"] = str(trace_files[0].name)

        except Exception as e:
            flame_data["error"] = str(e)
    else:
        flame_data["message"] = "No profile traces found. Run benchmarks with profiling enabled."

    return flame_data


def load_memory_timeline(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load memory usage timeline data."""
    memory_data: Dict[str, Any] = {
        "timeline": [],
        "peak_mb": 0,
        "summary": {
            "total_allocated_mb": 0,
            "peak_allocated_mb": 0,
            "num_allocations": 0,
        },
    }

    profile_dirs = [
        code_root / "benchmark_profiles",
        code_root / "artifacts" / "profiles",
    ]

    memory_files: List[Path] = []
    for profile_dir in profile_dirs:
        if profile_dir.exists():
            memory_files.extend(profile_dir.glob("**/*memory*.json"))
            memory_files.extend(profile_dir.glob("**/*memory*.pickle"))

    if memory_files:
        memory_files = sorted(memory_files, key=lambda f: f.stat().st_mtime, reverse=True)
        try:
            if memory_files[0].suffix == ".json":
                with open(memory_files[0]) as f:
                    data = json.load(f)
                memory_data.update(data)
                memory_data["has_real_data"] = True
        except Exception as e:
            memory_data["error"] = str(e)
    else:
        memory_data["has_real_data"] = False
        memory_data["message"] = "No memory profile data found. Run with: python -m torch.profiler"

    return memory_data


def load_cpu_gpu_timeline(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load CPU/GPU parallel timeline data."""
    timeline_data: Dict[str, Any] = {
        "cpu": [],
        "gpu": [],
        "streams": {},
        "summary": {
            "total_time_ms": 0,
            "cpu_time_ms": 0,
            "gpu_time_ms": 0,
            "overlap_ms": 0,
        },
    }

    profile_dirs = [
        code_root / "benchmark_profiles",
        code_root / "artifacts" / "profiles",
    ]

    trace_files: List[Path] = []
    for profile_dir in profile_dirs:
        if profile_dir.exists():
            trace_files.extend(profile_dir.glob("**/*.json"))

    if trace_files:
        trace_files = sorted(trace_files, key=lambda f: f.stat().st_mtime, reverse=True)
        try:
            with open(trace_files[0]) as f:
                trace = json.load(f)

            events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

            min_ts = float("inf")
            max_ts = 0

            for event in events[:500]:  # Limit for performance
                if event.get("ph") != "X":
                    continue

                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                name = event.get("name", "")
                cat = event.get("cat", "").lower()

                min_ts = min(min_ts, ts)
                max_ts = max(max_ts, ts + dur)

                event_data = {
                    "name": name[:50],
                    "start_ms": ts / 1000,
                    "duration_ms": dur / 1000,
                    # Attempt to propagate source info if present in trace
                    "file": event.get("file") or event.get("args", {}).get("file"),
                    "line": event.get("line") or event.get("args", {}).get("line"),
                    "pid": event.get("pid"),
                    "tid": event.get("tid"),
                    "cat": event.get("cat"),
                }

                if "cuda" in cat or "kernel" in cat:
                    timeline_data["gpu"].append(event_data)
                else:
                    timeline_data["cpu"].append(event_data)

            if max_ts > min_ts:
                timeline_data["summary"]["total_time_ms"] = (max_ts - min_ts) / 1000

        except Exception as e:
            timeline_data["error"] = str(e)

    return timeline_data


def load_kernel_breakdown(flame_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Get detailed kernel timing breakdown from flame graph data."""
    kernel_data: Dict[str, Any] = {
        "kernels": [],
        "summary": {
            "total_kernels": 0,
            "total_time_us": 0,
            "avg_kernel_time_us": 0,
        },
        "by_type": {},
    }

    flame = flame_data or load_flame_graph_data()

    all_kernels: List[Dict[str, Any]] = []
    for category in flame.get("children", []):
        cat_name = category.get("name", "other")
        for kernel in category.get("children", []):
            all_kernels.append(
                {
                    "name": kernel["name"],
                    "time_us": kernel["value"],
                    "category": cat_name,
                }
            )

            if cat_name not in kernel_data["by_type"]:
                kernel_data["by_type"][cat_name] = 0
            kernel_data["by_type"][cat_name] += kernel["value"]

    kernel_data["kernels"] = sorted(all_kernels, key=lambda k: -k["time_us"])[:50]
    kernel_data["summary"]["total_kernels"] = len(all_kernels)
    kernel_data["summary"]["total_time_us"] = sum(k["time_us"] for k in all_kernels)
    if all_kernels:
        kernel_data["summary"]["avg_kernel_time_us"] = (
            kernel_data["summary"]["total_time_us"] / len(all_kernels)
        )

    return kernel_data


def load_hta_analysis(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load HTA (Holistic Trace Analysis) results."""
    hta_data: Dict[str, Any] = {
        "temporal_breakdown": {
            "compute_pct": 70,
            "idle_pct": 15,
            "communication_pct": 10,
            "memory_pct": 5,
        },
        "top_kernels": [],
        "recommendations": [],
        "bottlenecks": [],
    }

    hta_files = list(code_root.glob("**/hta_report*.json"))
    hta_files.extend(code_root.glob("**/hta_analysis*.json"))

    if hta_files:
        try:
            with open(sorted(hta_files, key=lambda f: f.stat().st_mtime)[-1]) as f:
                data = json.load(f)
            hta_data.update(data)
        except Exception:
            hta_data["error"] = "Failed to parse HTA report"
    else:
        hta_data["message"] = "No HTA analysis found. Run HTA tooling to generate reports."

    return hta_data
