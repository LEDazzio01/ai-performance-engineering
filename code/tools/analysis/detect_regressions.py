"""Detect performance regressions from continuous benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Thresholds for different metrics
DEFAULT_THRESHOLDS = {
    "throughput_tokens_per_sec": {"direction": "higher_is_better", "threshold_pct": 5.0},
    "latency_ms": {"direction": "lower_is_better", "threshold_pct": 5.0},
    "memory_mb": {"direction": "lower_is_better", "threshold_pct": 10.0},
    "power_watts": {"direction": "lower_is_better", "threshold_pct": 10.0},
    "cost_per_million_tokens": {"direction": "lower_is_better", "threshold_pct": 5.0},
}


class RegressionDetector:
    """Detect regressions by comparing current and baseline metrics."""
    
    def __init__(self, thresholds: Optional[Dict] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
    
    def extract_metrics(self, benchmark_data: Dict) -> Dict[str, float]:
        """Extract numeric metrics from benchmark output."""
        metrics = {}
        # Handle different benchmark output formats
        if "benchmark_output" in benchmark_data:
            output = benchmark_data["benchmark_output"]
            # Try common metric names
            for key, value in output.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        return metrics
    
    def compare_metrics(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float]
    ) -> List[Dict]:
        """Compare current metrics against baseline.
        
        Returns list of regressions found.
        """
        regressions = []
        for metric_name, current_value in current.items():
            if metric_name not in baseline:
                continue
            
            baseline_value = baseline[metric_name]
            if baseline_value == 0:
                continue
            
            # Calculate percent change
            delta_pct = ((current_value - baseline_value) / baseline_value) * 100
            
            # Check if this metric has a threshold
            threshold_config = self.thresholds.get(metric_name, {
                "direction": "higher_is_better",
                "threshold_pct": 5.0
            })
            direction = threshold_config.get("direction", "higher_is_better")
            threshold = threshold_config.get("threshold_pct", 5.0)
            
            is_regression = False
            if direction == "higher_is_better":
                # Regression if current is lower than baseline by threshold
                if delta_pct < -threshold:
                    is_regression = True
            elif direction == "lower_is_better":
                # Regression if current is higher than baseline by threshold
                if delta_pct > threshold:
                    is_regression = True
            
            if is_regression:
                regressions.append({
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "delta_pct": delta_pct,
                    "threshold_pct": threshold,
                    "direction": direction,
                })
        
        return regressions
    
    def analyze_run(
        self,
        current_run: Dict,
        baseline_run: Dict
    ) -> Dict:
        """Analyze a benchmark run against baseline."""
        # Extract benchmark results
        current_benchmarks = current_run.get("benchmarks", [])
        baseline_benchmarks = baseline_run.get("benchmarks", [])
        
        # Map by benchmark name
        baseline_map = {
            b.get("name", "unknown"): b
            for b in baseline_benchmarks
        }
        
        results = {
            "current_timestamp": current_run.get("timestamp"),
            "baseline_timestamp": baseline_run.get("timestamp"),
            "regressions": [],
            "improvements": [],
            "stable": [],
        }
        
        for current_bench in current_benchmarks:
            bench_name = current_bench.get("name", "unknown")
            if bench_name not in baseline_map:
                continue
            
            baseline_bench = baseline_map[bench_name]
            current_metrics = self.extract_metrics(current_bench)
            baseline_metrics = self.extract_metrics(baseline_bench)
            
            regressions = self.compare_metrics(current_metrics, baseline_metrics)
            
            if regressions:
                results["regressions"].append({
                    "benchmark": bench_name,
                    "regressions": regressions,
                })
            else:
                # Check for improvements
                improvements = []
                for metric_name, current_value in current_metrics.items():
                    if metric_name not in baseline_metrics:
                        continue
                    
                    baseline_value = baseline_metrics[metric_name]
                    if baseline_value == 0:
                        continue
                    
                    delta_pct = ((current_value - baseline_value) / baseline_value) * 100
                    
                    threshold_config = self.thresholds.get(metric_name, {})
                    direction = threshold_config.get("direction", "higher_is_better")
                    
                    if direction == "higher_is_better" and delta_pct > 5.0:
                        improvements.append({
                            "metric": metric_name,
                            "baseline": baseline_value,
                            "current": current_value,
                            "delta_pct": delta_pct,
                        })
                    elif direction == "lower_is_better" and delta_pct < -5.0:
                        improvements.append({
                            "metric": metric_name,
                            "baseline": baseline_value,
                            "current": current_value,
                            "delta_pct": delta_pct,
                        })
                
                if improvements:
                    results["improvements"].append({
                        "benchmark": bench_name,
                        "improvements": improvements,
                    })
                else:
                    results["stable"].append(bench_name)
        
        return results


def load_benchmark_run(path: Path) -> Dict:
    """Load a benchmark run from JSON file."""
    return json.loads(path.read_text())


def find_latest_runs(artifact_dir: Path, n: int = 2) -> List[Path]:
    """Find the N most recent benchmark runs."""
    run_files = sorted(
        artifact_dir.glob("benchmark_run_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return run_files[:n]


def generate_regression_report(
    analysis: Dict,
    output_path: Optional[Path] = None
) -> str:
    """Generate a markdown report of regressions."""
    lines = []
    lines.append("# Performance Regression Report\n")
    lines.append(f"\n**Baseline**: {analysis.get('baseline_timestamp', 'unknown')}\n")
    lines.append(f"**Current**: {analysis.get('current_timestamp', 'unknown')}\n")
    lines.append("")
    
    # Summary
    num_regressions = len(analysis.get("regressions", []))
    num_improvements = len(analysis.get("improvements", []))
    num_stable = len(analysis.get("stable", []))
    
    lines.append("## Summary\n")
    lines.append(f"- 🚨 **Regressions**: {num_regressions}\n")
    lines.append(f"- ✅ **Improvements**: {num_improvements}\n")
    lines.append(f"- ➡️ **Stable**: {num_stable}\n")
    lines.append("")
    
    # Regressions
    if num_regressions > 0:
        lines.append("## 🚨 Regressions Detected\n")
        lines.append("")
        for item in analysis["regressions"]:
            bench_name = item["benchmark"]
            lines.append(f"### {bench_name}\n")
            lines.append("")
            lines.append("| Metric | Baseline | Current | Change | Threshold |\n")
            lines.append("|--------|----------|---------|--------|----------|\n")
            
            for reg in item["regressions"]:
                metric = reg["metric"]
                baseline = reg["baseline"]
                current = reg["current"]
                delta = reg["delta_pct"]
                threshold = reg["threshold_pct"]
                
                # Format values
                if abs(baseline) > 1000:
                    baseline_str = f"{baseline:,.0f}"
                    current_str = f"{current:,.0f}"
                else:
                    baseline_str = f"{baseline:.2f}"
                    current_str = f"{current:.2f}"
                
                change_str = f"{delta:+.1f}%"
                threshold_str = f"±{threshold:.0f}%"
                
                lines.append(
                    f"| {metric} | {baseline_str} | {current_str} | "
                    f"**{change_str}** | {threshold_str} |\n"
                )
            lines.append("")
    else:
        lines.append("## ✅ No Regressions Detected\n")
        lines.append("")
    
    # Improvements
    if num_improvements > 0:
        lines.append("## ✅ Performance Improvements\n")
        lines.append("")
        for item in analysis["improvements"]:
            bench_name = item["benchmark"]
            lines.append(f"### {bench_name}\n")
            lines.append("")
            lines.append("| Metric | Baseline | Current | Improvement |\n")
            lines.append("|--------|----------|---------|-------------|\n")
            
            for imp in item["improvements"]:
                metric = imp["metric"]
                baseline = imp["baseline"]
                current = imp["current"]
                delta = imp["delta_pct"]
                
                lines.append(
                    f"| {metric} | {baseline:.2f} | {current:.2f} | "
                    f"**{delta:+.1f}%** |\n"
                )
            lines.append("")
    
    # Stable benchmarks
    if num_stable > 0:
        lines.append("## ➡️ Stable Benchmarks\n")
        lines.append("")
        for bench_name in analysis["stable"]:
            lines.append(f"- {bench_name}\n")
        lines.append("")
    
    # Action items
    if num_regressions > 0:
        lines.append("## Action Items\n")
        lines.append("")
        lines.append("1. Review recent code changes that may have caused regressions\n")
        lines.append("2. Profile affected benchmarks to identify bottlenecks\n")
        lines.append("3. Consider reverting changes if regression is severe\n")
        lines.append("4. Update baseline if regression is expected and acceptable\n")
        lines.append("")
    
    report = "".join(lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Detect performance regressions from continuous benchmarks"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("benchmark_runs"),
        help="Directory containing benchmark runs"
    )
    parser.add_argument(
        "--current",
        type=Path,
        help="Current benchmark run JSON (default: latest)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline benchmark run JSON (default: second latest)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output markdown report (default: stdout)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON analysis file"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code if regressions detected"
    )
    
    args = parser.parse_args()
    
    # Find benchmark runs
    if args.current and args.baseline:
        current_path = args.current
        baseline_path = args.baseline
    else:
        runs = find_latest_runs(args.artifact_dir, n=2)
        if len(runs) < 2:
            print("Error: Need at least 2 benchmark runs to compare")
            return 1
        current_path = runs[0]
        baseline_path = runs[1]
    
    print(f"Comparing:")
    print(f"  Current: {current_path}")
    print(f"  Baseline: {baseline_path}")
    print()
    
    # Load runs
    current_run = load_benchmark_run(current_path)
    baseline_run = load_benchmark_run(baseline_path)
    
    # Analyze
    detector = RegressionDetector()
    analysis = detector.analyze_run(current_run, baseline_run)
    
    # Generate report
    report = generate_regression_report(analysis, args.output)
    
    # Print to stdout if no output file
    if not args.output:
        print(report)
    else:
        print(f"Report saved to: {args.output}")
    
    # Save JSON if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(analysis, indent=2))
        print(f"JSON analysis saved to: {args.output_json}")
    
    # Exit with error if regressions detected
    num_regressions = len(analysis.get("regressions", []))
    if args.fail_on_regression and num_regressions > 0:
        print(f"\nERROR: {num_regressions} regression(s) detected!")
        return 1
    
    if num_regressions > 0:
        print(f"\nWARNING: {num_regressions} regression(s) detected")
    else:
        print("\n✅ No regressions detected")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
