"""Calculate cost per token metrics from power and throughput data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


# Default energy costs (USD per kWh) - update based on your region
DEFAULT_ELECTRICITY_COST = {
    "us_average": 0.16,       # US national average
    "us_industrial": 0.08,    # US industrial rate
    "europe_average": 0.25,   # Europe average
    "cloud_premium": 0.30,    # Typical cloud markup
}

# Hardware efficiency estimates (PUE - Power Usage Effectiveness)
DEFAULT_PUE = {
    "best_in_class": 1.1,     # Google, Meta level datacenters
    "enterprise": 1.5,        # Typical enterprise datacenter
    "cloud": 1.7,             # Typical cloud provider
}


def load_power_metrics(power_json: Path) -> Dict:
    """Load power metrics from power_monitor.py output."""
    data = json.loads(power_json.read_text())
    return data


def load_throughput_metrics(throughput_json: Path) -> Dict:
    """Load throughput metrics from benchmark output."""
    data = json.loads(throughput_json.read_text())
    return data


def calculate_cost_per_token(
    avg_power_watts: float,
    throughput_tokens_per_sec: float,
    electricity_cost_per_kwh: float,
    pue: float = 1.5,
) -> Dict[str, float]:
    """Calculate cost per token and related metrics.
    
    Args:
        avg_power_watts: Average power consumption in watts
        throughput_tokens_per_sec: Throughput in tokens/second
        electricity_cost_per_kwh: Cost of electricity per kWh
        pue: Power Usage Effectiveness (datacenter overhead)
    
    Returns:
        Dict with cost metrics
    """
    
    if throughput_tokens_per_sec <= 0:
        return {
            "error": "Invalid throughput (must be > 0)"
        }
    
    # Account for datacenter overhead
    total_power_watts = avg_power_watts * pue
    
    # Convert to kW
    power_kw = total_power_watts / 1000.0
    
    # Cost per hour
    cost_per_hour = power_kw * electricity_cost_per_kwh
    
    # Cost per second
    cost_per_second = cost_per_hour / 3600.0
    
    # Cost per token
    cost_per_token = cost_per_second / throughput_tokens_per_sec
    
    # Cost per 1M tokens
    cost_per_million_tokens = cost_per_token * 1_000_000
    
    # Tokens per dollar
    tokens_per_dollar = 1.0 / cost_per_token if cost_per_token > 0 else 0
    
    # Energy per token (joules)
    energy_per_token = total_power_watts / throughput_tokens_per_sec
    
    return {
        "avg_power_watts": avg_power_watts,
        "total_power_watts_with_pue": total_power_watts,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "electricity_cost_per_kwh": electricity_cost_per_kwh,
        "pue": pue,
        "cost_per_token_usd": cost_per_token,
        "cost_per_million_tokens_usd": cost_per_million_tokens,
        "tokens_per_dollar": tokens_per_dollar,
        "energy_per_token_joules": energy_per_token,
        "cost_per_hour_usd": cost_per_hour,
    }


def generate_cost_report(
    workload_name: str,
    metrics: Dict,
    electricity_scenarios: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a detailed cost analysis report."""
    
    if electricity_scenarios is None:
        electricity_scenarios = DEFAULT_ELECTRICITY_COST
    
    lines = []
    lines.append("# Cost Per Token Analysis\n")
    lines.append(f"\n**Workload**: {workload_name}\n")
    lines.append("")
    
    # Base metrics
    lines.append("## Base Metrics\n")
    lines.append(f"- **Average Power**: {metrics['avg_power_watts']:.1f} W\n")
    lines.append(f"- **Power with PUE ({metrics['pue']}x)**: {metrics['total_power_watts_with_pue']:.1f} W\n")
    lines.append(f"- **Throughput**: {metrics['throughput_tokens_per_sec']:.1f} tokens/sec\n")
    lines.append(f"- **Energy per Token**: {metrics['energy_per_token_joules']:.4f} J\n")
    lines.append("")
    
    # Cost analysis
    lines.append("## Cost Analysis\n")
    lines.append(f"- **Electricity Rate**: ${metrics['electricity_cost_per_kwh']:.3f}/kWh\n")
    lines.append(f"- **Cost per Token**: ${metrics['cost_per_token_usd']:.8f}\n")
    lines.append(f"- **Cost per Million Tokens**: ${metrics['cost_per_million_tokens_usd']:.2f}\n")
    lines.append(f"- **Tokens per Dollar**: {metrics['tokens_per_dollar']:,.0f}\n")
    lines.append(f"- **Operating Cost**: ${metrics['cost_per_hour_usd']:.2f}/hour\n")
    lines.append("")
    
    # Scenario comparison
    lines.append("## Electricity Cost Scenarios\n")
    lines.append("\n| Scenario | $/kWh | $/token | $/1M tokens | Tokens/$ |\n")
    lines.append("|----------|-------|---------|-------------|----------|\n")
    
    for scenario_name, elec_cost in electricity_scenarios.items():
        scenario_metrics = calculate_cost_per_token(
            metrics['avg_power_watts'],
            metrics['throughput_tokens_per_sec'],
            elec_cost,
            metrics['pue']
        )
        lines.append(
            f"| {scenario_name.replace('_', ' ').title()} | "
            f"${elec_cost:.3f} | "
            f"${scenario_metrics['cost_per_token_usd']:.8f} | "
            f"${scenario_metrics['cost_per_million_tokens_usd']:.2f} | "
            f"{scenario_metrics['tokens_per_dollar']:,.0f} |\n"
        )
    
    lines.append("")
    
    # Efficiency insights
    lines.append("## Efficiency Insights\n")
    lines.append("")
    
    # Compare to typical API pricing
    typical_api_cost_per_million = 2.00  # Typical API cost per 1M tokens
    cost_ratio = metrics['cost_per_million_tokens_usd'] / typical_api_cost_per_million
    
    lines.append(f"### Cost Comparison\n")
    lines.append(f"- **Your cost per 1M tokens**: ${metrics['cost_per_million_tokens_usd']:.2f}\n")
    lines.append(f"- **Typical API cost per 1M tokens**: ${typical_api_cost_per_million:.2f}\n")
    lines.append(f"- **Ratio**: {cost_ratio:.2f}x\n")
    lines.append("")
    
    if cost_ratio < 0.1:
        lines.append("✅ **Excellent**: Energy costs are minimal compared to API pricing\n")
    elif cost_ratio < 0.5:
        lines.append("✅ **Good**: Energy costs are low, plenty of margin for hardware amortization\n")
    elif cost_ratio < 1.0:
        lines.append("⚠️ **Moderate**: Energy costs are significant but below API pricing\n")
    else:
        lines.append("🚨 **High**: Energy costs exceed typical API pricing - check efficiency\n")
    
    lines.append("")
    
    # Break-even analysis
    lines.append("### Break-even Analysis\n")
    lines.append("")
    
    # Example hardware costs
    b200_cost = 40000  # Approximate cost of B200 GPU
    server_cost = 60000  # Full server cost estimate
    
    # Calculate tokens needed to break even (energy only)
    tokens_to_break_even = server_cost / metrics['cost_per_token_usd']
    
    # Calculate time to break even at current throughput
    hours_to_break_even = tokens_to_break_even / (metrics['throughput_tokens_per_sec'] * 3600)
    days_to_break_even = hours_to_break_even / 24
    
    lines.append(f"Assuming hardware cost: ${server_cost:,}\n")
    lines.append(f"- **Tokens to break even** (energy only): {tokens_to_break_even:,.0f}\n")
    lines.append(f"- **Time to break even** (at full utilization): {days_to_break_even:.1f} days\n")
    lines.append("")
    lines.append("*Note: This is energy cost only. Real break-even includes hardware depreciation, cooling, maintenance, etc.*\n")
    
    lines.append("")
    lines.append("## Optimization Recommendations\n")
    lines.append("")
    
    if metrics['energy_per_token_joules'] > 10:
        lines.append("1. ⚠️ High energy per token - consider:\n")
        lines.append("   - Batch size optimization\n")
        lines.append("   - Model quantization (FP8)\n")
        lines.append("   - Better GPU utilization\n")
    else:
        lines.append("1. ✅ Energy efficiency looks good\n")
    
    if metrics['pue'] > 1.5:
        lines.append(f"2. ⚠️ PUE of {metrics['pue']} is high - datacenter efficiency could be improved\n")
    else:
        lines.append(f"2. ✅ PUE of {metrics['pue']} is reasonable\n")
    
    lines.append("")
    
    report = "".join(lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Cost report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Calculate cost per token from power and throughput data"
    )
    parser.add_argument(
        "--power-json",
        type=Path,
        help="Power metrics JSON from power_monitor.py"
    )
    parser.add_argument(
        "--throughput-json",
        type=Path,
        help="Throughput metrics JSON from benchmark"
    )
    parser.add_argument(
        "--workload-name",
        default="Benchmark",
        help="Name of the workload"
    )
    parser.add_argument(
        "--avg-power",
        type=float,
        help="Manual average power in watts (alternative to --power-json)"
    )
    parser.add_argument(
        "--throughput",
        type=float,
        help="Manual throughput in tokens/sec (alternative to --throughput-json)"
    )
    parser.add_argument(
        "--electricity-cost",
        type=float,
        default=0.16,
        help="Electricity cost in USD per kWh (default: 0.16)"
    )
    parser.add_argument(
        "--pue",
        type=float,
        default=1.5,
        help="Power Usage Effectiveness (default: 1.5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output markdown report file"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON metrics file"
    )
    args = parser.parse_args()
    
    # Get power metrics
    if args.power_json:
        power_data = load_power_metrics(args.power_json)
        avg_power = power_data["total_power"]["avg_watts"]
    elif args.avg_power:
        avg_power = args.avg_power
    else:
        print("Error: Either --power-json or --avg-power is required")
        return 1
    
    # Get throughput metrics
    if args.throughput_json:
        throughput_data = load_throughput_metrics(args.throughput_json)
        # Try to extract throughput from common benchmark formats
        if "tokens_per_sec" in throughput_data:
            throughput = throughput_data["tokens_per_sec"]
        elif "throughput" in throughput_data:
            throughput = throughput_data["throughput"]
        else:
            print("Error: Could not find throughput in JSON")
            print("Available keys:", list(throughput_data.keys()))
            return 1
    elif args.throughput:
        throughput = args.throughput
    else:
        print("Error: Either --throughput-json or --throughput is required")
        return 1
    
    # Calculate metrics
    metrics = calculate_cost_per_token(
        avg_power,
        throughput,
        args.electricity_cost,
        args.pue
    )
    
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return 1
    
    # Generate report
    report = generate_cost_report(
        args.workload_name,
        metrics,
        output_path=args.output
    )
    
    # Print to stdout if no output file
    if not args.output:
        print(report)
    
    # Save JSON if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics saved to: {args.output_json}")
    
    return 0


if __name__ == "__main__":
    exit(main())


