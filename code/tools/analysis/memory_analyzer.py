"""Analyzes memory profiling results and generates summary reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_memory_profile(file_path: Path) -> Dict[str, float]:
    """Extract key memory metrics from a memory profiling text output."""
    metrics = {
        "peak_cuda_mb": 0.0,
        "total_cuda_mb": 0.0,
        "top_operations": []
    }
    
    try:
        content = file_path.read_text()
        
        # Look for memory usage in the table
        # Pattern: operation name, self cpu time, cuda time, cpu memory, cuda memory
        memory_pattern = r'(\d+\.\d+)\s+[KMG]b'
        
        # Find all memory values
        memory_values = []
        for line in content.split('\n'):
            if 'Mb' in line or 'Gb' in line or 'Kb' in line:
                # Extract memory values
                matches = re.findall(r'([\d.]+)\s+([KMG]b)', line)
                for value, unit in matches:
                    value_float = float(value)
                    if unit == 'Gb':
                        value_float *= 1024
                    elif unit == 'Kb':
                        value_float /= 1024
                    memory_values.append(value_float)
        
        if memory_values:
            metrics["peak_cuda_mb"] = max(memory_values)
            metrics["total_cuda_mb"] = sum(memory_values)
        
        # Extract top operations from the table
        in_table = False
        for line in content.split('\n'):
            if '----' in line and ('Self CPU' in content or 'CUDA Mem' in content):
                in_table = True
                continue
            if in_table and line.strip():
                # Try to extract operation name and memory
                parts = line.split()
                if len(parts) > 0 and not parts[0].startswith('-'):
                    # Operation name is typically the first part
                    op_name = parts[0]
                    metrics["top_operations"].append(op_name)
                if len(metrics["top_operations"]) >= 5:
                    break
    
    except Exception as e:
        print(f"Warning: Could not fully parse {file_path}: {e}")
    
    return metrics


def generate_summary(memory_dir: Path) -> str:
    """Generate a markdown summary from all memory profiles."""
    
    summary_lines = []
    summary_lines.append("# Memory Profiling Summary\n")
    summary_lines.append(f"**Analyzed**: {memory_dir.name}\n")
    summary_lines.append("")
    
    # Find all .txt files
    profile_files = sorted(memory_dir.glob("*.txt"))
    
    if not profile_files:
        summary_lines.append("⚠️ No memory profile files found.\n")
        return "\n".join(summary_lines)
    
    summary_lines.append("## Workload Memory Usage\n")
    summary_lines.append("| Workload | Peak CUDA Memory | Total Allocated |\n")
    summary_lines.append("|----------|------------------|------------------|\n")
    
    total_peak = 0.0
    workload_data = []
    
    for profile_file in profile_files:
        workload_name = profile_file.stem.replace("_memory", "").replace("_", " ").title()
        metrics = parse_memory_profile(profile_file)
        
        peak_mb = metrics["peak_cuda_mb"]
        total_mb = metrics["total_cuda_mb"]
        
        total_peak += peak_mb
        workload_data.append((workload_name, peak_mb, total_mb))
        
        summary_lines.append(
            f"| {workload_name} | {peak_mb:.1f} MB | {total_mb:.1f} MB |\n"
        )
    
    summary_lines.append("")
    summary_lines.append(f"**Total Peak Memory Across Workloads**: {total_peak:.1f} MB ({total_peak/1024:.2f} GB)\n")
    summary_lines.append("")
    
    # Find the most memory-intensive workload
    if workload_data:
        most_intensive = max(workload_data, key=lambda x: x[1])
        summary_lines.append(f"**Most Memory-Intensive**: {most_intensive[0]} ({most_intensive[1]:.1f} MB peak)\n")
    
    summary_lines.append("")
    summary_lines.append("## Analysis\n")
    summary_lines.append("")
    
    # Add context about B200 memory capacity
    b200_memory_gb = 180
    if total_peak > 0:
        utilization = (total_peak / 1024) / b200_memory_gb * 100
        summary_lines.append(f"- B200 HBM3e Capacity: {b200_memory_gb} GB\n")
        summary_lines.append(f"- Peak Utilization: {utilization:.1f}%\n")
        
        if utilization < 20:
            summary_lines.append("- Status: ✅ Low memory usage - workloads fit comfortably\n")
        elif utilization < 60:
            summary_lines.append("- Status: ✅ Moderate memory usage - room for larger models\n")
        elif utilization < 90:
            summary_lines.append("- Status: ⚠️ High memory usage - approaching capacity\n")
        else:
            summary_lines.append("- Status: 🚨 Very high memory usage - risk of OOM\n")
    
    summary_lines.append("")
    summary_lines.append("## Chrome Traces\n")
    summary_lines.append("")
    
    # Check for Chrome traces
    trace_files = sorted(memory_dir.glob("*.json"))
    if trace_files:
        summary_lines.append("Chrome traces available for detailed analysis:\n")
        for trace_file in trace_files:
            summary_lines.append(f"- `{trace_file.name}`\n")
        summary_lines.append("\nLoad these in `chrome://tracing` for timeline visualization.\n")
    else:
        summary_lines.append("No Chrome traces generated.\n")
    
    return "".join(summary_lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze memory profiling results")
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing memory profile outputs")
    parser.add_argument("--output", type=Path,
                        help="Output file for summary (default: stdout)")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        return 1
    
    summary = generate_summary(args.input)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
        print(f"Memory summary written to {args.output}")
    else:
        print(summary)
    
    return 0


if __name__ == "__main__":
    exit(main())


