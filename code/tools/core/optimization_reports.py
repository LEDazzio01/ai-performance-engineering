"""
Shared optimization ROI and summary helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_roi(benchmarks: List[Dict[str, Any]], cost_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate ROI for each optimization technique based on benchmark data and cost model."""
    technique_roi: Dict[str, Dict[str, Any]] = {}

    for b in benchmarks:
        for opt in b.get("optimizations", []):
            technique = opt.get("technique", opt.get("file", "unknown"))
            speedup = opt.get("speedup", 1.0)

            if technique not in technique_roi:
                technique_roi[technique] = {
                    "count": 0,
                    "total_speedup": 0.0,
                    "benchmarks": [],
                }

            technique_roi[technique]["count"] += 1
            technique_roi[technique]["total_speedup"] += speedup
            technique_roi[technique]["benchmarks"].append(b.get("name", ""))

    roi_results = []
    hourly_rate = cost_data.get("hourly_rate", 5.00)

    for technique, data in technique_roi.items():
        avg_speedup = data["total_speedup"] / data["count"] if data["count"] > 0 else 1.0

        ops_per_month = 720 * 1_000_000  # assume 1M ops/hour for rough ROI
        time_saved_pct = (1 - 1 / avg_speedup) * 100 if avg_speedup > 1 else 0
        monthly_savings = hourly_rate * 720 * (time_saved_pct / 100)

        effort = "low" if any(x in technique.lower() for x in ["compile", "tf32", "cudnn"]) else "medium"
        if any(x in technique.lower() for x in ["triton", "cuda", "kernel"]):
            effort = "high"

        roi_results.append(
            {
                "technique": technique,
                "avg_speedup": round(avg_speedup, 2),
                "benchmarks_using": data["count"],
                "time_saved_pct": round(time_saved_pct, 1),
                "monthly_savings_usd": round(monthly_savings, 2),
                "annual_savings_usd": round(monthly_savings * 12, 2),
                "implementation_effort": effort,
                "roi_score": round(
                    monthly_savings / (10 if effort == "low" else 50 if effort == "medium" else 100), 2
                ),
            }
        )

    roi_results.sort(key=lambda x: x["roi_score"], reverse=True)

    return {
        "techniques": roi_results,
        "summary": {
            "total_techniques": len(roi_results),
            "total_monthly_savings": round(
                sum(r["monthly_savings_usd"] for r in roi_results), 2
            ),
            "best_roi_technique": roi_results[0]["technique"] if roi_results else None,
        },
    }

