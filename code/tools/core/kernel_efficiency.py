"""
Shared kernel efficiency scoring using simple heuristics vs theoretical peaks.
"""

from __future__ import annotations

from typing import Any, Dict, List


def score_kernels(kernel_data: Dict[str, Any]) -> Dict[str, Any]:
    """Score kernels against heuristic peak efficiencies."""
    peaks = {
        "bf16_tflops": 2500.0,
        "fp8_tflops": 5000.0,
        "fp32_tflops": 625.0,
        "hbm_bandwidth_gbs": 8000.0,
        "tensor_core_utilization": 100.0,
    }

    kernel_efficiency: List[Dict[str, Any]] = []
    for kernel in kernel_data.get("kernels", [])[:20]:
        k_name = kernel.get("name", "")
        k_time_us = kernel.get("time_us", 0)
        k_lower = k_name.lower()

        if "gemm" in k_lower or "matmul" in k_lower:
            flops_eff = 65.0
            mem_eff = 40.0
            tc_util = 70.0
            ktype = "GEMM"
        elif "conv" in k_lower:
            flops_eff = 55.0
            mem_eff = 50.0
            tc_util = 60.0
            ktype = "Convolution"
        elif "attention" in k_lower or "sdpa" in k_lower:
            flops_eff = 45.0
            mem_eff = 70.0
            tc_util = 50.0
            ktype = "Attention"
        elif "softmax" in k_lower:
            flops_eff = 20.0
            mem_eff = 80.0
            tc_util = 0.0
            ktype = "Softmax"
        elif "layernorm" in k_lower or "norm" in k_lower:
            flops_eff = 15.0
            mem_eff = 85.0
            tc_util = 0.0
            ktype = "Normalization"
        elif "copy" in k_lower or "memcpy" in k_lower:
            flops_eff = 0.0
            mem_eff = 90.0
            tc_util = 0.0
            ktype = "Memory Copy"
        else:
            flops_eff = 30.0
            mem_eff = 50.0
            tc_util = 20.0
            ktype = "Other"

        flops_gap = peaks["bf16_tflops"] * (1 - flops_eff / 100)
        mem_gap = peaks["hbm_bandwidth_gbs"] * (1 - mem_eff / 100)

        kernel_efficiency.append(
            {
                "name": k_name[:60],
                "type": ktype,
                "time_us": k_time_us,
                "flops_efficiency_pct": round(flops_eff, 1),
                "memory_efficiency_pct": round(mem_eff, 1),
                "tensor_core_utilization_pct": round(tc_util, 1),
                "flops_gap_tflops": round(flops_gap, 0),
                "memory_gap_gbs": round(mem_gap, 0),
                "bottleneck": "compute" if flops_eff > mem_eff else "memory",
                "optimization_potential": "high"
                if max(flops_eff, mem_eff) < 50
                else "medium"
                if max(flops_eff, mem_eff) < 75
                else "low",
            }
        )

    type_summary: Dict[str, Dict[str, Any]] = {}
    for k in kernel_efficiency:
        kt = k["type"]
        if kt not in type_summary:
            type_summary[kt] = {"count": 0, "total_time": 0, "avg_flops_eff": 0.0, "avg_mem_eff": 0.0}
        type_summary[kt]["count"] += 1
        type_summary[kt]["total_time"] += k["time_us"]
        type_summary[kt]["avg_flops_eff"] += k["flops_efficiency_pct"]
        type_summary[kt]["avg_mem_eff"] += k["memory_efficiency_pct"]

    for kt in type_summary:
        type_summary[kt]["avg_flops_eff"] /= type_summary[kt]["count"]
        type_summary[kt]["avg_mem_eff"] /= type_summary[kt]["count"]

    overall = {
        "avg_flops_efficiency": round(
            sum(k["flops_efficiency_pct"] for k in kernel_efficiency) / len(kernel_efficiency), 1
        )
        if kernel_efficiency
        else 0,
        "avg_memory_efficiency": round(
            sum(k["memory_efficiency_pct"] for k in kernel_efficiency) / len(kernel_efficiency), 1
        )
        if kernel_efficiency
        else 0,
        "avg_tc_utilization": round(
            sum(k["tensor_core_utilization_pct"] for k in kernel_efficiency) / len(kernel_efficiency), 1
        )
        if kernel_efficiency
        else 0,
    }

    return {
        "peaks": peaks,
        "kernels": kernel_efficiency,
        "by_type": type_summary,
        "overall": overall,
    }

