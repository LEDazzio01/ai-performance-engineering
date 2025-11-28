#!/usr/bin/env python3
"""
Pareto Frontier Analysis

Analyzes cost/throughput/memory tradeoffs and identifies Pareto-optimal
configurations for distributed training/inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class ConfigurationPoint:
    """A single configuration with its metrics."""
    
    # Configuration
    name: str
    tp: int
    pp: int
    dp: int
    
    # Metrics (required)
    throughput_tps: float  # Tokens per second
    latency_ms: float
    memory_per_gpu_gb: float
    num_gpus: int
    
    # Optional configuration
    cp: int = 1
    ep: int = 1
    sharding: str = "none"
    
    # Cost metrics
    gpu_hours_per_million_tokens: float = 0.0
    cost_per_million_tokens: float = 0.0
    
    # Derived
    efficiency: float = 0.0  # Throughput / num_gpus
    
    def __post_init__(self):
        if self.num_gpus > 0 and self.throughput_tps > 0:
            self.efficiency = self.throughput_tps / self.num_gpus
            # GPU hours to process 1M tokens
            tokens_per_gpu_hour = self.throughput_tps * 3600 / self.num_gpus
            if tokens_per_gpu_hour > 0:
                self.gpu_hours_per_million_tokens = 1e6 / tokens_per_gpu_hour
    
    @property
    def config_str(self) -> str:
        parts = [f"TP={self.tp}", f"PP={self.pp}", f"DP={self.dp}"]
        if self.cp > 1:
            parts.append(f"CP={self.cp}")
        if self.ep > 1:
            parts.append(f"EP={self.ep}")
        if self.sharding != "none":
            parts.append(self.sharding.upper())
        return "×".join(parts[:3]) + ("+" + "+".join(parts[3:]) if len(parts) > 3 else "")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config_str,
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "cp": self.cp,
            "ep": self.ep,
            "sharding": self.sharding,
            "throughput_tps": self.throughput_tps,
            "latency_ms": self.latency_ms,
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "num_gpus": self.num_gpus,
            "efficiency": self.efficiency,
            "gpu_hours_per_million_tokens": self.gpu_hours_per_million_tokens,
            "cost_per_million_tokens": self.cost_per_million_tokens,
        }


@dataclass
class ParetoFrontier:
    """Pareto-optimal configurations for a given objective pair."""
    
    x_metric: str  # e.g., "cost", "latency", "memory"
    y_metric: str  # e.g., "throughput"
    
    # All configurations analyzed
    all_points: List[ConfigurationPoint]
    
    # Pareto-optimal configurations
    pareto_points: List[ConfigurationPoint]
    
    # Best for specific objectives
    best_throughput: Optional[ConfigurationPoint] = None
    best_efficiency: Optional[ConfigurationPoint] = None
    best_latency: Optional[ConfigurationPoint] = None
    lowest_memory: Optional[ConfigurationPoint] = None
    lowest_cost: Optional[ConfigurationPoint] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_metric": self.x_metric,
            "y_metric": self.y_metric,
            "num_configurations": len(self.all_points),
            "num_pareto_optimal": len(self.pareto_points),
            "pareto_points": [p.to_dict() for p in self.pareto_points],
            "best_throughput": self.best_throughput.to_dict() if self.best_throughput else None,
            "best_efficiency": self.best_efficiency.to_dict() if self.best_efficiency else None,
            "best_latency": self.best_latency.to_dict() if self.best_latency else None,
            "lowest_memory": self.lowest_memory.to_dict() if self.lowest_memory else None,
            "lowest_cost": self.lowest_cost.to_dict() if self.lowest_cost else None,
        }


# GPU pricing ($/hour) - approximate cloud rates
GPU_PRICING = {
    "A100_40GB": 1.50,
    "A100_80GB": 2.00,
    "H100_80GB": 3.50,
    "H100_SXM": 4.00,
    "B200": 6.00,  # Estimated
    "GB200": 8.00,  # Estimated
}


class ParetoAnalyzer:
    """Analyzes Pareto frontiers for parallelism configurations."""
    
    def __init__(self, gpu_hourly_cost: float = 4.00):
        """Initialize analyzer.
        
        Args:
            gpu_hourly_cost: Cost per GPU hour in dollars
        """
        self.gpu_hourly_cost = gpu_hourly_cost
    
    def set_gpu_pricing(self, gpu_type: str) -> None:
        """Set GPU pricing by type."""
        if gpu_type in GPU_PRICING:
            self.gpu_hourly_cost = GPU_PRICING[gpu_type]
    
    def analyze(
        self,
        configurations: List[ConfigurationPoint],
        x_metric: str = "num_gpus",
        y_metric: str = "throughput_tps",
    ) -> ParetoFrontier:
        """Analyze configurations and find Pareto frontier.
        
        Args:
            configurations: List of configurations to analyze
            x_metric: X-axis metric (to minimize)
            y_metric: Y-axis metric (to maximize)
            
        Returns:
            ParetoFrontier with analysis results
        """
        if not configurations:
            return ParetoFrontier(
                x_metric=x_metric,
                y_metric=y_metric,
                all_points=[],
                pareto_points=[],
            )
        
        # Calculate cost for all configurations
        for config in configurations:
            config.cost_per_million_tokens = (
                config.gpu_hours_per_million_tokens * self.gpu_hourly_cost
            )
        
        # Find Pareto-optimal points
        pareto_points = self._find_pareto_frontier(configurations, x_metric, y_metric)
        
        # Find best for each objective
        best_throughput = max(configurations, key=lambda c: c.throughput_tps)
        best_efficiency = max(configurations, key=lambda c: c.efficiency)
        best_latency = min(configurations, key=lambda c: c.latency_ms if c.latency_ms > 0 else float('inf'))
        lowest_memory = min(configurations, key=lambda c: c.memory_per_gpu_gb)
        lowest_cost = min(configurations, key=lambda c: c.cost_per_million_tokens if c.cost_per_million_tokens > 0 else float('inf'))
        
        return ParetoFrontier(
            x_metric=x_metric,
            y_metric=y_metric,
            all_points=configurations,
            pareto_points=pareto_points,
            best_throughput=best_throughput,
            best_efficiency=best_efficiency,
            best_latency=best_latency if best_latency.latency_ms > 0 else None,
            lowest_memory=lowest_memory,
            lowest_cost=lowest_cost if lowest_cost.cost_per_million_tokens > 0 else None,
        )
    
    def _find_pareto_frontier(
        self,
        points: List[ConfigurationPoint],
        x_metric: str,
        y_metric: str,
    ) -> List[ConfigurationPoint]:
        """Find Pareto-optimal points (minimize x, maximize y)."""
        
        def get_metric(point: ConfigurationPoint, metric: str) -> float:
            return getattr(point, metric, 0)
        
        pareto = []
        
        for candidate in points:
            x_val = get_metric(candidate, x_metric)
            y_val = get_metric(candidate, y_metric)
            
            is_dominated = False
            for other in points:
                if other is candidate:
                    continue
                
                other_x = get_metric(other, x_metric)
                other_y = get_metric(other, y_metric)
                
                # other dominates candidate if:
                # - other_x <= x_val AND other_y >= y_val
                # - AND at least one strict inequality
                if (other_x <= x_val and other_y >= y_val and
                    (other_x < x_val or other_y > y_val)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(candidate)
        
        # Sort by x metric
        pareto.sort(key=lambda p: get_metric(p, x_metric))
        
        return pareto
    
    def generate_cost_throughput_analysis(
        self,
        configurations: List[ConfigurationPoint],
    ) -> Dict[str, Any]:
        """Generate comprehensive cost/throughput analysis.
        
        Returns:
            Analysis dictionary with recommendations
        """
        if not configurations:
            return {"error": "No configurations to analyze"}
        
        # Calculate metrics
        for config in configurations:
            config.cost_per_million_tokens = (
                config.gpu_hours_per_million_tokens * self.gpu_hourly_cost
            )
        
        # Find frontiers for different objectives
        cost_throughput = self.analyze(configurations, "cost_per_million_tokens", "throughput_tps")
        gpu_throughput = self.analyze(configurations, "num_gpus", "throughput_tps")
        memory_throughput = self.analyze(configurations, "memory_per_gpu_gb", "throughput_tps")
        
        # Generate recommendations
        recommendations = []
        
        if cost_throughput.lowest_cost:
            recommendations.append({
                "objective": "Minimize Cost",
                "config": cost_throughput.lowest_cost.config_str,
                "cost_per_m_tokens": f"${cost_throughput.lowest_cost.cost_per_million_tokens:.2f}",
                "throughput": f"{cost_throughput.lowest_cost.throughput_tps:.0f} tps",
            })
        
        if cost_throughput.best_throughput:
            recommendations.append({
                "objective": "Maximize Throughput",
                "config": cost_throughput.best_throughput.config_str,
                "cost_per_m_tokens": f"${cost_throughput.best_throughput.cost_per_million_tokens:.2f}",
                "throughput": f"{cost_throughput.best_throughput.throughput_tps:.0f} tps",
            })
        
        if cost_throughput.best_efficiency:
            recommendations.append({
                "objective": "Best Efficiency (tps/GPU)",
                "config": cost_throughput.best_efficiency.config_str,
                "efficiency": f"{cost_throughput.best_efficiency.efficiency:.0f} tps/GPU",
                "throughput": f"{cost_throughput.best_efficiency.throughput_tps:.0f} tps",
            })
        
        # Find "sweet spot" - good balance of cost and throughput
        pareto = cost_throughput.pareto_points
        if len(pareto) >= 3:
            # Pick middle of Pareto frontier
            sweet_spot = pareto[len(pareto) // 2]
            recommendations.append({
                "objective": "Balanced (Sweet Spot)",
                "config": sweet_spot.config_str,
                "cost_per_m_tokens": f"${sweet_spot.cost_per_million_tokens:.2f}",
                "throughput": f"{sweet_spot.throughput_tps:.0f} tps",
            })
        
        return {
            "gpu_hourly_cost": self.gpu_hourly_cost,
            "num_configurations": len(configurations),
            "recommendations": recommendations,
            "pareto_frontier": {
                "cost_vs_throughput": [p.to_dict() for p in cost_throughput.pareto_points],
                "gpus_vs_throughput": [p.to_dict() for p in gpu_throughput.pareto_points],
            },
            "extremes": {
                "best_throughput": cost_throughput.best_throughput.to_dict() if cost_throughput.best_throughput else None,
                "lowest_cost": cost_throughput.lowest_cost.to_dict() if cost_throughput.lowest_cost else None,
                "best_efficiency": cost_throughput.best_efficiency.to_dict() if cost_throughput.best_efficiency else None,
            },
        }
    
    def format_pareto_report(
        self,
        configurations: List[ConfigurationPoint],
    ) -> str:
        """Generate human-readable Pareto analysis report."""
        analysis = self.generate_cost_throughput_analysis(configurations)
        
        lines = [
            "=" * 80,
            "COST / THROUGHPUT PARETO ANALYSIS",
            "=" * 80,
            "",
            f"GPU Hourly Cost: ${self.gpu_hourly_cost:.2f}",
            f"Configurations Analyzed: {analysis['num_configurations']}",
            "",
            "RECOMMENDATIONS:",
            "-" * 40,
        ]
        
        for rec in analysis.get("recommendations", []):
            lines.extend([
                "",
                f"  {rec['objective']}:",
                f"    Config: {rec['config']}",
            ])
            for key, value in rec.items():
                if key not in ("objective", "config"):
                    lines.append(f"    {key.replace('_', ' ').title()}: {value}")
        
        pareto = analysis.get("pareto_frontier", {}).get("cost_vs_throughput", [])
        if pareto:
            lines.extend([
                "",
                "PARETO FRONTIER (Cost vs Throughput):",
                "-" * 40,
                "",
                f"{'Config':<25} {'Throughput':>12} {'Cost/M tokens':>15} {'GPUs':>6}",
                "-" * 60,
            ])
            for p in pareto[:10]:  # Show top 10
                lines.append(
                    f"{p['config']:<25} {p['throughput_tps']:>10,.0f} tps "
                    f"${p['cost_per_million_tokens']:>12.2f} {p['num_gpus']:>6}"
                )
        
        lines.extend(["", "=" * 80])
        
        return "\n".join(lines)
    
    def generate_visualization_data(
        self,
        configurations: List[ConfigurationPoint],
    ) -> Dict[str, Any]:
        """Generate data suitable for visualization (charts, graphs).
        
        Returns:
            Data structure for frontend visualization
        """
        analysis = self.generate_cost_throughput_analysis(configurations)
        
        # Prepare scatter plot data
        scatter_data = [
            {
                "x": c.cost_per_million_tokens,
                "y": c.throughput_tps,
                "label": c.config_str,
                "num_gpus": c.num_gpus,
                "is_pareto": c.config_str in [
                    p["config"] for p in analysis.get("pareto_frontier", {}).get("cost_vs_throughput", [])
                ],
            }
            for c in configurations
            if c.cost_per_million_tokens > 0
        ]
        
        # Pareto frontier line
        pareto_line = [
            {"x": p["cost_per_million_tokens"], "y": p["throughput_tps"]}
            for p in analysis.get("pareto_frontier", {}).get("cost_vs_throughput", [])
        ]
        
        return {
            "scatter_data": scatter_data,
            "pareto_line": pareto_line,
            "x_axis": {
                "label": "Cost per Million Tokens ($)",
                "min": 0,
                "max": max(c.cost_per_million_tokens for c in configurations) * 1.1 if configurations else 10,
            },
            "y_axis": {
                "label": "Throughput (tokens/s)",
                "min": 0,
                "max": max(c.throughput_tps for c in configurations) * 1.1 if configurations else 100000,
            },
            "recommendations": analysis.get("recommendations", []),
        }

