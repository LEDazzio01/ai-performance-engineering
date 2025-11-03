#!/usr/bin/env python3
"""
Power efficiency analyzer for AI workloads.
Calculates tokens/joule, cost/token, and energy efficiency metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


def load_power_metrics(power_file: Path) -> Dict[str, Any]:
    """Load power monitoring data."""
    with open(power_file) as f:
        data = json.load(f)
    
    # Check if it's the aggregate format from power_monitor.py
    if "total_power" in data and "energy_joules" in data:
        total_power = data["total_power"]
        return {
            "avg_power_w": total_power["avg_watts"],
            "max_power_w": total_power["max_watts"],
            "min_power_w": total_power["min_watts"],
            "duration_s": data["duration"],
            "total_energy_j": data["energy_joules"],
        }
    elif "samples" in data and isinstance(data["samples"], list):
        # Detailed samples format
        samples = data["samples"]
        power_watts = [s["power_watts"] if "power_watts" in s else s.get("total_watts", 0) for s in samples]
        timestamps = [s["timestamp"] for s in samples]
        
        return {
            "avg_power_w": np.mean(power_watts),
            "max_power_w": np.max(power_watts),
            "min_power_w": np.min(power_watts),
            "duration_s": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            "total_energy_j": np.trapz(power_watts, timestamps) if len(timestamps) > 1 else 0,
        }
    else:
        # Legacy format
        return data


def load_throughput_metrics(throughput_file: Path) -> Dict[str, Any]:
    """Load throughput/performance data."""
    with open(throughput_file) as f:
        data = json.load(f)
    
    # Try to extract relevant metrics
    metrics = {}
    
    if "tokens_per_second" in data:
        metrics["tokens_per_second"] = data["tokens_per_second"]
    elif "throughput_tokens_per_sec" in data:
        metrics["tokens_per_second"] = data["throughput_tokens_per_sec"]
    elif "throughput_tok_s" in data:
        metrics["tokens_per_second"] = data["throughput_tok_s"]
    elif "throughput" in data:
        metrics["tokens_per_second"] = data["throughput"]
    
    if "total_tokens" in data:
        metrics["total_tokens"] = data["total_tokens"]
    
    if "latency_ms" in data:
        metrics["latency_ms"] = data["latency_ms"]
    elif "avg_latency_ms" in data:
        metrics["latency_ms"] = data["avg_latency_ms"]
    
    if "batch_size" in data:
        metrics["batch_size"] = data["batch_size"]
    elif "config" in data and "batch_size" in data["config"]:
        metrics["batch_size"] = data["config"]["batch_size"]
    
    if "sequence_length" in data:
        metrics["sequence_length"] = data["sequence_length"]
    elif "config" in data and "seq_len" in data["config"]:
        metrics["sequence_length"] = data["config"]["seq_len"]
    
    return metrics


def calculate_power_efficiency(
    power_data: Dict[str, Any],
    throughput_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate power efficiency metrics.
    
    Returns:
        Dictionary with efficiency metrics:
        - tokens_per_joule: Tokens generated per joule of energy
        - cost_per_million_tokens: Estimated cost (assuming $0.50/kWh)
        - efficiency_score: Normalized efficiency metric
    """
    results = {}
    
    # Extract key metrics
    avg_power_w = power_data.get("avg_power_w", 0)
    total_energy_j = power_data.get("total_energy_j", 0)
    duration_s = power_data.get("duration_s", 0)
    
    tokens_per_sec = throughput_data.get("tokens_per_second", 0)
    total_tokens = throughput_data.get("total_tokens")
    
    # Calculate total tokens if not provided
    if total_tokens is None and tokens_per_sec > 0 and duration_s > 0:
        total_tokens = tokens_per_sec * duration_s
    
    # Calculate tokens per joule
    if total_energy_j > 0 and total_tokens:
        tokens_per_joule = total_tokens / total_energy_j
        results["tokens_per_joule"] = tokens_per_joule
        
        # Calculate cost per million tokens (assuming $0.50/kWh)
        kwh_per_token = (1 / tokens_per_joule) / 3600 / 1000
        cost_per_million_tokens = kwh_per_token * 1_000_000 * 0.50
        results["cost_per_million_tokens_usd"] = cost_per_million_tokens
    
    # Calculate power per token
    if avg_power_w > 0 and tokens_per_sec > 0:
        joules_per_token = avg_power_w / tokens_per_sec
        results["joules_per_token"] = joules_per_token
    
    # Add raw metrics for reference
    results["avg_power_watts"] = avg_power_w
    results["max_power_watts"] = power_data.get("max_power_w", 0)
    results["total_energy_joules"] = total_energy_j
    results["duration_seconds"] = duration_s
    results["tokens_per_second"] = tokens_per_sec
    results["total_tokens"] = total_tokens
    
    # Calculate efficiency score (tokens/joule normalized to 0-100 scale)
    # Reference: ~1 token/joule is good for large models
    if "tokens_per_joule" in results:
        results["efficiency_score"] = min(100, results["tokens_per_joule"] * 100)
    
    return results


def generate_efficiency_report(efficiency_data: Dict[str, Any]) -> str:
    """Generate markdown report for power efficiency."""
    lines = []
    
    lines.append("# Power Efficiency Analysis Report")
    lines.append("")
    lines.append("## Energy Consumption")
    lines.append("")
    lines.append(f"- **Average Power**: {efficiency_data.get('avg_power_watts', 0):.2f} W")
    lines.append(f"- **Max Power**: {efficiency_data.get('max_power_watts', 0):.2f} W")
    lines.append(f"- **Total Energy**: {efficiency_data.get('total_energy_joules', 0):.2f} J")
    lines.append(f"- **Duration**: {efficiency_data.get('duration_seconds', 0):.2f} s")
    lines.append("")
    
    lines.append("## Throughput")
    lines.append("")
    lines.append(f"- **Tokens/Second**: {efficiency_data.get('tokens_per_second', 0):.2f}")
    lines.append(f"- **Total Tokens**: {efficiency_data.get('total_tokens', 0):,}")
    lines.append("")
    
    lines.append("## Efficiency Metrics")
    lines.append("")
    
    if "tokens_per_joule" in efficiency_data:
        lines.append(f"- **Tokens per Joule**: {efficiency_data['tokens_per_joule']:.4f}")
        lines.append(f"- **Joules per Token**: {efficiency_data.get('joules_per_token', 0):.4f}")
        lines.append(f"- **Cost per Million Tokens**: ${efficiency_data.get('cost_per_million_tokens_usd', 0):.4f}")
        lines.append(f"- **Efficiency Score**: {efficiency_data.get('efficiency_score', 0):.1f}/100")
    else:
        lines.append("- **Insufficient data for efficiency calculations**")
    
    lines.append("")
    
    # Add interpretation
    lines.append("## Interpretation")
    lines.append("")
    
    if "tokens_per_joule" in efficiency_data:
        tpj = efficiency_data["tokens_per_joule"]
        if tpj > 1.0:
            lines.append("✅ **Excellent** efficiency (>1 token/joule)")
        elif tpj > 0.5:
            lines.append("✅ **Good** efficiency (0.5-1 token/joule)")
        elif tpj > 0.1:
            lines.append("⚠️ **Moderate** efficiency (0.1-0.5 token/joule)")
        else:
            lines.append("❌ **Low** efficiency (<0.1 token/joule)")
    
    lines.append("")
    lines.append("---")
    lines.append("*Reference: State-of-art inference can achieve 1-2 tokens/joule on modern hardware*")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze power efficiency of AI workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze inference power efficiency
  python tools/power_efficiency_analyzer.py \\
      --power power_metrics.json \\
      --throughput inference_results.json \\
      --output efficiency_report.md
  
  # Quick analysis with JSON output only
  python tools/power_efficiency_analyzer.py \\
      --power power_metrics.json \\
      --throughput inference_results.json \\
      --json-output efficiency.json
        """
    )
    
    parser.add_argument(
        "--power-file", "--power",
        type=Path,
        required=True,
        help="Path to power monitoring JSON file"
    )
    
    parser.add_argument(
        "--throughput-file", "--throughput",
        type=Path,
        required=True,
        help="Path to throughput/performance JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output markdown report file"
    )
    
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Output JSON file with efficiency metrics"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.power_file.exists():
        print(f"Error: Power file not found: {args.power_file}", file=sys.stderr)
        return 1
    
    if not args.throughput_file.exists():
        print(f"Error: Throughput file not found: {args.throughput_file}", file=sys.stderr)
        return 1
    
    try:
        # Load data
        print(f"Loading power metrics from: {args.power_file}")
        power_data = load_power_metrics(args.power_file)
        
        print(f"Loading throughput metrics from: {args.throughput_file}")
        throughput_data = load_throughput_metrics(args.throughput_file)
        
        # Calculate efficiency
        print("Calculating power efficiency...")
        efficiency_data = calculate_power_efficiency(power_data, throughput_data)
        
        # Write JSON output if requested
        if args.json_output:
            print(f"Writing JSON to: {args.json_output}")
            with open(args.json_output, "w") as f:
                json.dump(efficiency_data, f, indent=2)
        
        # Generate markdown report
        report = generate_efficiency_report(efficiency_data)
        
        # Write or print report
        if args.output:
            print(f"Writing report to: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
        else:
            print("\n" + report)
        
        # Print key metrics to console
        print("\nKey Metrics:")
        print(f"  Tokens/Joule: {efficiency_data.get('tokens_per_joule', 0):.4f}")
        print(f"  Cost/Million Tokens: ${efficiency_data.get('cost_per_million_tokens_usd', 0):.4f}")
        print(f"  Efficiency Score: {efficiency_data.get('efficiency_score', 0):.1f}/100")
        
        return 0
    
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

