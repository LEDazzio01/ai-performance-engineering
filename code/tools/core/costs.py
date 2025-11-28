"""
Shared cost calculator and TCO estimator utilities.
"""

from __future__ import annotations

from typing import Dict, Any, List


GPU_PRICING = {
    "B200": 5.00,
    "H100": 3.50,
    "A100": 2.00,
    "L40S": 1.50,
    "A10G": 1.00,
    "T4": 0.50,
}


def detect_gpu_pricing(gpu_name: str) -> float:
    current_gpu = "B200"
    for g in GPU_PRICING:
        if g in gpu_name:
            current_gpu = g
            break
    return GPU_PRICING.get(current_gpu, 5.00)


def calculate_costs(benchmarks: List[Dict[str, Any]], gpu_info: Dict[str, Any]) -> Dict[str, Any]:
    """Compute cost per million operations and savings for benchmarks."""
    gpu_name = gpu_info.get("name", "B200")
    current_rate = detect_gpu_pricing(gpu_name)

    cost_analysis = []
    for b in benchmarks[:20]:
        baseline_ms = b.get("baseline_time_ms", 0)
        optimized_ms = b.get("optimized_time_ms", baseline_ms)
        speedup = b.get("speedup", 1.0)

        if baseline_ms > 0:
            baseline_ops_per_hour = 3_600_000 / baseline_ms
            optimized_ops_per_hour = 3_600_000 / optimized_ms if optimized_ms > 0 else baseline_ops_per_hour

            baseline_cost_per_million = (current_rate / baseline_ops_per_hour) * 1_000_000
            optimized_cost_per_million = (current_rate / optimized_ops_per_hour) * 1_000_000

            savings_per_million = baseline_cost_per_million - optimized_cost_per_million

            cost_analysis.append(
                {
                    "name": b.get("name"),
                    "chapter": b.get("chapter"),
                    "speedup": speedup,
                    "baseline_cost_per_million": baseline_cost_per_million,
                    "optimized_cost_per_million": optimized_cost_per_million,
                    "savings_per_million": savings_per_million,
                }
            )

    return {
        "gpu_pricing": GPU_PRICING,
        "current_gpu": gpu_name,
        "current_rate": current_rate,
        "benchmarks": cost_analysis,
    }

