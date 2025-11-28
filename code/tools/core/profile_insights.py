"""
Shared profiling insights and heuristics used across dashboard, CLI, and MCP.

These helpers stay free of HTTP/handler state and operate on plain data
structures so they can be reused by any interface.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_ncu_sources(ncu_path: Path, limit: int = 12) -> List[Dict[str, str]]:
    """
    Best-effort extraction of source/line info from an ncu-rep.
    Parses the Source page CSV and keeps the first few entries with a Source column.
    """
    try:
        result = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "source"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0:
            return []
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        kernel_name: Optional[str] = None
        header: Optional[List[str]] = None
        samples: List[Dict[str, str]] = []
        for row in csv.reader(lines):
            if not row:
                continue
            if row[0] == "Kernel Name":
                kernel_name = row[1] if len(row) > 1 else kernel_name
                header = None
                continue
            if "Source" in row and "Address" in row:
                header = row
                continue
            if header and len(row) >= len(header):
                entry = dict(zip(header, row))
                source = entry.get("Source", "").strip()
                if not source:
                    continue
                samples.append(
                    {
                        "kernel": kernel_name or "",
                        "source": source,
                        "address": entry.get("Address", "").strip(),
                        "stall": entry.get("Warp Stall Sampling (All Samples)", "").strip(),
                    }
                )
                if len(samples) >= limit:
                    break
        return samples
    except Exception:
        return []


def _extract_ncu_disassembly(ncu_path: Path, limit: int = 40) -> List[str]:
    """
    Best-effort disassembly from an ncu-rep via the SASS page.
    Returns a list of text lines (truncated) for quick inspection.
    """
    try:
        result = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "sass"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0:
            return []
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        # Keep only meaningful lines (skip headers and empty)
        payload: List[str] = []
        for ln in lines:
            if ln.startswith("==") or ln.startswith("Kernel Name"):
                continue
            payload.append(ln)
            if len(payload) >= limit:
                break
        return payload
    except Exception:
        return []


def _extract_nsys_sources(nsys_path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    """
    Best-effort extraction of source/line info from an nsys-rep by exporting trace JSON.
    Broadened to capture CPU/GPU/NVTX/driver events with any file/line/symbol hints.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="nsys_src_") as tmp:
            out_prefix = Path(tmp) / "trace"
            export_cmd = [
                "nsys",
                "export",
                "--type",
                "timeline",  # richer than stats; still JSON
                "--format",
                "json",
                "--force",
                "true",
                "--output",
                str(out_prefix),
                str(nsys_path),
            ]
            subprocess.run(export_cmd, check=True, capture_output=True, text=True, timeout=90)
            json_path = out_prefix.with_suffix(".json")
            if not json_path.exists():
                return []
            data = json.loads(json_path.read_text())
            events = data if isinstance(data, list) else data.get("traceEvents", [])
            samples: List[Dict[str, Any]] = []
            seen = set()
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                cat = (ev.get("cat") or "").lower()
                # Broader cat coverage: cuda, gpu, cpu, nvtx, driver/cu/rt, cupti
                if not any(tok in cat for tok in ("cuda", "gpu", "cpu", "nvtx", "driver", "cupti")):
                    continue
                args = ev.get("args") or {}
                file = ev.get("file") or args.get("file") or args.get("Source File")
                line = ev.get("line") or args.get("line") or args.get("Source Line")
                symbol = args.get("External Symbol") or args.get("Symbol") or ev.get("name")
                key = (file, line, symbol, ev.get("pid"), ev.get("tid"))
                if key in seen:
                    continue
                seen.add(key)
                samples.append(
                    {
                        "name": ev.get("name", ""),
                        "cat": ev.get("cat", ""),
                        "file": file,
                        "line": line,
                        "symbol": symbol,
                        "ts": ev.get("ts"),
                        "dur": ev.get("dur"),
                        "pid": ev.get("pid"),
                        "tid": ev.get("tid"),
                    }
                )
                if len(samples) >= limit:
                    break
            return samples
    except Exception:
        return []


def detect_bottlenecks(
    flame_data: Optional[Dict[str, Any]],
    kernel_data: Optional[Dict[str, Any]],
    hw_caps: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze profile data to detect performance bottlenecks."""
    flame_data = flame_data or {}
    kernel_data = kernel_data or {}
    hw_caps = hw_caps or {}

    bottlenecks: List[Dict[str, Any]] = []
    total_time = float(flame_data.get("value") or 0)
    if total_time <= 0:
        return {"bottlenecks": [], "message": "No profile data available"}

    features = hw_caps.get("features", []) or []
    feature_names = [f.get("name", "") for f in features if isinstance(f, dict)]

    # Analyze by category
    for category in flame_data.get("children", []):
        cat_name = category.get("name", "")
        cat_time = float(category.get("value") or 0)
        cat_pct = (cat_time / total_time * 100) if total_time else 0.0

        # Memory transfer bottleneck
        if cat_name in ["gpu_memcpy", "cuda_runtime"] and cat_pct > 10:
            bottlenecks.append(
                {
                    "type": "memory_transfer",
                    "severity": "high" if cat_pct > 20 else "medium",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Memory transfers taking {cat_pct:.1f}% of execution time",
                    "suggestions": [
                        "Use pinned memory for host-to-device transfers",
                        "Overlap transfers with compute using CUDA streams",
                        "Consider using TMA for async memory operations"
                        if any(name.startswith("TMA") for name in feature_names)
                        else "Batch small transfers into larger ones",
                        "Use cudaMemcpyAsync instead of synchronous copies",
                    ],
                    "potential_speedup": f"{cat_pct * 0.5:.0f}% reduction possible",
                }
            )

        # CPU overhead bottleneck
        if cat_name == "python_function" and cat_pct > 15:
            bottlenecks.append(
                {
                    "type": "cpu_overhead",
                    "severity": "high" if cat_pct > 30 else "medium",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Python/CPU overhead taking {cat_pct:.1f}% of execution time",
                    "suggestions": [
                        "Use torch.compile() to fuse operations",
                        "Enable CUDA graphs for repeated kernel launches",
                        "Reduce Python-level loops - vectorize operations",
                        "Use torch.jit.script for hot paths",
                    ],
                    "potential_speedup": f"{cat_pct * 0.7:.0f}% reduction possible",
                }
            )

        # Profiler overhead (informational)
        if cat_name == "overhead" and cat_pct > 5:
            bottlenecks.append(
                {
                    "type": "profiler_overhead",
                    "severity": "info",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Profiler overhead is {cat_pct:.1f}% - actual performance is better",
                    "suggestions": [
                        "This overhead is from profiling only",
                        f"Real execution will be faster by ~{cat_pct:.0f}%",
                    ],
                    "potential_speedup": "N/A - profiling artifact",
                }
            )

    # Analyze individual kernels for optimization opportunities
    kernels = kernel_data.get("kernels", []) or []
    total_kernel_time = float(kernel_data.get("summary", {}).get("total_time_us", 1) or 1)

    for kernel in kernels[:5]:  # Top 5 kernels
        k_name = kernel.get("name", "")
        k_time = float(kernel.get("time_us", 0) or 0)
        k_pct = (k_time / total_kernel_time * 100) if total_kernel_time else 0.0

        if k_pct > 15:  # Significant kernel
            suggestions: List[str] = []
            lower_name = k_name.lower()

            # Pattern-based suggestions
            if any(term in lower_name for term in ["gemm", "matmul", "mm_"]):
                suggestions.extend(
                    [
                        "Ensure matrix dimensions are multiples of 16",
                        "Consider using cuBLAS or cuBLASLt for GEMM",
                        "Try FP8 for inference if accuracy permits"
                        if (hw_caps.get("architecture") or "").lower().startswith("blackwell")
                        else "Use TF32 for faster FP32 operations",
                    ]
                )
            elif "conv" in lower_name:
                suggestions.extend(
                    [
                        "Use cuDNN with autotuning enabled",
                        "Consider channels-last memory format (NHWC)",
                        "Try depthwise-separable convolutions if applicable",
                    ]
                )
            elif any(term in lower_name for term in ["softmax", "attention", "sdpa"]):
                suggestions.extend(
                    [
                        "Use Flash Attention for memory efficiency",
                        "Consider fused attention kernels",
                        "Use scaled_dot_product_attention with efficient backend",
                    ]
                )
            elif any(term in lower_name for term in ["copy", "memcpy", "_to_"]):
                suggestions.extend(
                    [
                        "Minimize data movement between CPU and GPU",
                        "Use in-place operations where possible",
                        "Consider memory layout optimization",
                    ]
                )
            else:
                suggestions.extend(
                    [
                        "Profile with NCU for detailed metrics",
                        "Check occupancy and register usage",
                        "Consider kernel fusion opportunities",
                    ]
                )

            bottlenecks.append(
                {
                    "type": "hot_kernel",
                    "severity": "high" if k_pct > 25 else "medium",
                    "category": "kernel",
                    "kernel_name": k_name,
                    "time_us": k_time,
                    "percentage": round(k_pct, 1),
                    "description": f"Kernel '{k_name[:50]}' taking {k_pct:.1f}% of GPU time",
                    "suggestions": suggestions,
                    "potential_speedup": "Varies by optimization",
                }
            )

    severity_order = {"high": 0, "medium": 1, "info": 2}
    bottlenecks.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 3))

    return {
        "bottlenecks": bottlenecks,
        "total_time_us": total_time,
        "hardware": {
            "name": hw_caps.get("gpu", {}).get("name", "Unknown"),
            "architecture": hw_caps.get("architecture", "unknown"),
            "features_available": len(features),
        },
    }


def calculate_optimization_score(
    hw_caps: Dict[str, Any], bottlenecks: Dict[str, Any], kernel_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate an optimization opportunity score based on profile analysis."""
    hw_caps = hw_caps or {}
    bottlenecks = bottlenecks or {}
    kernel_data = kernel_data or {}

    score = 100  # Start with perfect score
    deductions: List[str] = []
    quick_wins: List[Dict[str, Any]] = []
    advanced_opts: List[Dict[str, Any]] = []
    already_optimized: List[str] = []

    # Check for bottlenecks and deduct points
    for bottleneck in bottlenecks.get("bottlenecks", []) or []:
        severity = bottleneck.get("severity")
        if severity == "high":
            score -= 15
            deductions.append(f"{bottleneck.get('type')}: -15 pts")
        elif severity == "medium":
            score -= 8
            deductions.append(f"{bottleneck.get('type')}: -8 pts")

    # Check for unused hardware features
    for feature in hw_caps.get("features", []) or []:
        if not isinstance(feature, dict) or not feature.get("supported"):
            continue
        name = feature.get("name", "")
        opt_text = feature.get("optimization", "")
        if "TMA" in name:
            quick_wins.append(
                {
                    "feature": name,
                    "potential": "+15% speedup",
                    "effort": "medium",
                    "description": opt_text,
                }
            )
        elif "FP8" in name:
            quick_wins.append(
                {
                    "feature": name,
                    "potential": "+25% throughput",
                    "effort": "low",
                    "description": opt_text,
                }
            )
        elif "Cluster" in name:
            advanced_opts.append(
                {
                    "feature": name,
                    "potential": "+10% speedup",
                    "effort": "high",
                    "description": opt_text,
                }
            )
        elif "DSMEM" in name:
            advanced_opts.append(
                {
                    "feature": name,
                    "potential": "+5-15% speedup",
                    "effort": "high",
                    "description": opt_text,
                }
            )

    # Check what's already optimized (look for patterns in kernel names)
    kernels = kernel_data.get("kernels", []) or []
    kernel_names = [k.get("name", "").lower() for k in kernels]

    if any("flash" in k for k in kernel_names):
        already_optimized.append("Flash Attention enabled")
        score += 5
    if any("fused" in k for k in kernel_names):
        already_optimized.append("Kernel fusion detected")
        score += 3
    if any("cudnn" in k for k in kernel_names):
        already_optimized.append("cuDNN optimized kernels")
        score += 3

    score = max(0, min(100, score))  # Clamp to 0-100

    return {
        "score": score,
        "grade": "A"
        if score >= 90
        else "B"
        if score >= 75
        else "C"
        if score >= 60
        else "D"
        if score >= 40
        else "F",
        "deductions": deductions,
        "quick_wins": quick_wins,
        "advanced_optimizations": advanced_opts,
        "already_optimized": already_optimized,
        "summary": {
            "quick_win_count": len(quick_wins),
            "advanced_opt_count": len(advanced_opts),
            "already_optimized_count": len(already_optimized),
        },
    }


def compare_nsys_files(profiles_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract and compare nsys metrics between baseline and optimized."""
    baseline_nsys = list(profiles_dir.glob("*baseline*.nsys-rep"))
    optimized_nsys = list(profiles_dir.glob("*optimized*.nsys-rep"))

    if not baseline_nsys or not optimized_nsys:
        return None

    try:
        from tools.profiling.extract_nsys_summary import harvest

        baseline_metrics = harvest(baseline_nsys[0])
        optimized_metrics = harvest(optimized_nsys[0])

        comparison = {
            "baseline_file": baseline_nsys[0].name,
            "optimized_file": optimized_nsys[0].name,
            "metrics": [],
            "baseline_sources": _extract_nsys_sources(baseline_nsys[0]),
            "optimized_sources": _extract_nsys_sources(optimized_nsys[0]),
        }

        opt_lookup = {m.get("metric", ""): m.get("value", "") for m in optimized_metrics}

        for base_metric in baseline_metrics:
            metric = base_metric.get("metric", "")
            baseline_val = base_metric.get("value", "")
            optimized_val = opt_lookup.get(metric, "")

            delta = None
            delta_pct = None
            try:
                b_num = float(str(baseline_val).replace(",", ""))
                o_num = float(str(optimized_val).replace(",", ""))
                delta = o_num - b_num
                delta_pct = ((o_num - b_num) / b_num * 100) if b_num != 0 else 0
            except (ValueError, TypeError):
                pass

            comparison["metrics"].append(
                {
                    "name": metric,
                    "baseline": baseline_val,
                    "optimized": optimized_val,
                    "delta": delta,
                    "delta_pct": delta_pct,
                }
            )

        return comparison
    except Exception as exc:  # pragma: no cover - best-effort helper
        return {"error": str(exc)}


def compare_ncu_files(profiles_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract and compare ncu metrics between baseline and optimized."""
    baseline_ncu = list(profiles_dir.glob("*baseline*.ncu-rep"))
    optimized_ncu = list(profiles_dir.glob("*optimized*.ncu-rep"))

    baseline_csv = list(profiles_dir.glob("*baseline*ncu*.csv"))
    optimized_csv = list(profiles_dir.glob("*optimized*ncu*.csv"))

    if baseline_csv and optimized_csv:
        try:
            def read_ncu_csv(path: Path) -> Dict[str, Any]:
                metrics: Dict[str, Any] = {}
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("Metric Name", row.get("Name", ""))
                        value = row.get("Metric Value", row.get("Avg", row.get("Value", "")))
                        if name:
                            metrics[name] = value
                return metrics

            baseline_metrics = read_ncu_csv(baseline_csv[0])
            optimized_metrics = read_ncu_csv(optimized_csv[0])

            comparison = {
                "baseline_file": baseline_csv[0].name,
                "optimized_file": optimized_csv[0].name,
                "metrics": [],
            }

            all_keys = set(baseline_metrics.keys()) | set(optimized_metrics.keys())
            for key in sorted(all_keys):
                b_val = baseline_metrics.get(key, "")
                o_val = optimized_metrics.get(key, "")

                delta = None
                try:
                    b_num = float(str(b_val).replace(",", ""))
                    o_num = float(str(o_val).replace(",", ""))
                    delta = o_num - b_num
                except (ValueError, TypeError):
                    pass

                comparison["metrics"].append(
                    {"name": key, "baseline": b_val, "optimized": o_val, "delta": delta}
                )

            return comparison
        except Exception as exc:
            return {"error": str(exc)}

    if not baseline_ncu or not optimized_ncu:
        return None

    try:
        def extract_ncu_metrics(ncu_path: Path) -> Dict[str, Any]:
            result = subprocess.run(
                ["ncu", "--import", str(ncu_path), "--csv"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return {}

            metrics: Dict[str, Any] = {}
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "," in line and not line.startswith("=="):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        metrics[parts[0]] = parts[1] if len(parts) > 1 else ""
                return metrics

        baseline_metrics = extract_ncu_metrics(baseline_ncu[0])
        optimized_metrics = extract_ncu_metrics(optimized_ncu[0])

        if baseline_metrics or optimized_metrics:
            return {
                "baseline_file": baseline_ncu[0].name,
                "optimized_file": optimized_ncu[0].name,
                "baseline_metrics": baseline_metrics,
                "optimized_metrics": optimized_metrics,
                "baseline_sources": _extract_ncu_sources(baseline_ncu[0]),
                "optimized_sources": _extract_ncu_sources(optimized_ncu[0]),
                "baseline_disassembly": _extract_ncu_disassembly(baseline_ncu[0]),
                "optimized_disassembly": _extract_ncu_disassembly(optimized_ncu[0]),
            }
    except Exception as exc:  # pragma: no cover - tool availability varies
        return {"error": f"NCU extraction failed: {exc}"}

    return None


def generate_recommendations_from_profiles(result: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on profile comparison data."""
    recommendations: List[str] = []

    nsys = result.get("nsys_comparison", {})
    if nsys and isinstance(nsys, dict) and "metrics" in nsys:
        for metric in nsys.get("metrics", []):
            name = metric.get("name", "").lower()
            delta = metric.get("delta")

            if delta and "dram" in name and delta < -10:
                recommendations.append(
                    "Memory traffic reduced: Consider further optimization with TMA prefetching"
                )
            elif delta and "sm" in name and delta > 10:
                recommendations.append(
                    f"SM utilization improved by {delta:.1f}%: Good progress on compute efficiency"
                )

    ncu = result.get("ncu_comparison", {})
    if ncu and isinstance(ncu, dict) and "metrics" in ncu:
        for metric in ncu.get("metrics", []):
            name = metric.get("name", "").lower()
            if "occupancy" in name:
                try:
                    opt_val = float(str(metric.get("optimized", "0")).replace("%", ""))
                    if opt_val < 50:
                        recommendations.append(
                            f"Occupancy at {opt_val:.0f}%: Consider tuning block size or reducing register pressure"
                        )
                except (ValueError, TypeError):
                    pass

    if not recommendations:
        recommendations.append("Profile both baseline and optimized to get detailed recommendations")

    return recommendations
