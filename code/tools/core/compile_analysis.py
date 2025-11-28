"""
Shared helpers for torch.compile analysis based on benchmark artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

CODE_ROOT = Path(__file__).parent.parent.parent


def load_compile_analysis(code_root: Path = CODE_ROOT, benchmarks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Aggregate compile analysis from saved reports and benchmark data."""
    compile_data: Dict[str, Any] = {
        "speedup": 0,
        "compile_time_ms": 0,
        "graph_breaks": 0,
        "fusion_ratio": 1.0,
        "recommendations": [],
        "mode_comparison": {},
        "graph_breaks_list": [],
        "compile_benchmarks": [],
        "has_real_data": False,
    }

    compile_files = list(code_root.glob("**/compile_report*.json"))
    compile_files.extend(code_root.glob("**/torch_compile*.json"))
    compile_files.extend(code_root.glob("compile_analysis/*.json"))

    if compile_files:
        try:
            with open(sorted(compile_files, key=lambda f: f.stat().st_mtime)[-1]) as f:
                data = json.load(f)
            compile_data.update(data)
            compile_data["has_real_data"] = True
        except Exception:
            pass

    # Always compute from benchmark data if available
    if benchmarks is not None:
        compiled_benchmarks = []
        for b in benchmarks:
            name = str(b.get("name", "")).lower()
            chapter = str(b.get("chapter", "")).lower()
            techniques = str(b.get("optimizations", [])).lower()

            is_compile = any(
                [
                    "compile" in name,
                    "model_eager" in name,
                    "torch_compile" in techniques,
                    "inductor" in techniques,
                    chapter == "ch14" and ("model" in name or "eager" in name),
                ]
            )

            if is_compile and b.get("speedup", 0) > 0:
                compiled_benchmarks.append(
                    {
                        "name": b.get("name"),
                        "chapter": b.get("chapter"),
                        "speedup": b.get("speedup", 1.0),
                        "baseline_time_ms": b.get("baseline_time_ms", 0),
                        "optimized_time_ms": b.get("optimized_time_ms", 0),
                    }
                )

        compile_data["compile_benchmarks"] = compiled_benchmarks

        if compiled_benchmarks:
            speedups = [b["speedup"] for b in compiled_benchmarks if b["speedup"] > 0]
            if speedups:
                compile_data["speedup"] = sum(speedups) / len(speedups)
                compile_data["has_real_data"] = True
                total_baseline_ms = sum(b.get("baseline_time_ms", 0) for b in compiled_benchmarks)
                compile_data["compile_time_ms"] = total_baseline_ms * 0.1  # rough heuristic

                compile_data["recommendations"] = [
                    f"torch.compile achieved {compile_data['speedup']:.2f}x avg speedup across {len(compiled_benchmarks)} benchmarks",
                    f"Best compile speedup: {max(speedups):.2f}x ({[b for b in compiled_benchmarks if b['speedup'] == max(speedups)][0]['name']})",
                ]

                if compile_data["speedup"] < 1.5:
                    compile_data["recommendations"].append("Consider mode='max-autotune' for better optimization")
            else:
                compile_data.setdefault("recommendations", []).append(
                    "Compile benchmarks found but no speedup recorded."
                )
        else:
            compile_data["recommendations"] = [
                "No torch.compile benchmarks found. Run benchmarks in ch14 or use torch.compile in your code."
            ]

    return compile_data

