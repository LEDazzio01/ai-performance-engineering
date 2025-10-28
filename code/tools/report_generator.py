#!/usr/bin/env python3
"""
Markdown report generator for performance analysis results.
Generates reports similar to docs/performance_baseline.md format.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from performance_targets import (
    TARGETS,
    get_chapter_description,
    get_chapter_metrics,
    get_all_chapters,
    compute_status,
    format_value,
)


class PerformanceReport:
    """Generate markdown performance analysis reports."""
    
    def __init__(self):
        self.chapters_analyzed: List[str] = []
        self.metrics_by_chapter: Dict[str, Dict[str, Any]] = {}
        self.summary_stats = {
            "total_metrics": 0,
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "missing_count": 0,
        }
    
    def add_chapter_metrics(self, chapter: str, metrics: Dict[str, float]):
        """
        Add metrics for a chapter.
        
        Args:
            chapter: Chapter identifier (e.g., "ch2")
            metrics: Dictionary of metric_name -> value
        """
        if chapter not in self.chapters_analyzed:
            self.chapters_analyzed.append(chapter)
        
        # Compare against targets
        chapter_targets = get_chapter_metrics(chapter)
        results = {}
        
        for metric_name, target_def in chapter_targets.items():
            actual_value = metrics.get(metric_name)
            
            if actual_value is not None:
                status = compute_status(actual_value, target_def)
                results[metric_name] = {
                    "actual": actual_value,
                    "target": target_def,
                    "status": status,
                }
                
                # Update summary stats
                self.summary_stats["total_metrics"] += 1
                if status == "PASS":
                    self.summary_stats["pass_count"] += 1
                elif status == "WARN":
                    self.summary_stats["warn_count"] += 1
                elif status == "FAIL":
                    self.summary_stats["fail_count"] += 1
            else:
                # Metric not found
                results[metric_name] = {
                    "actual": None,
                    "target": target_def,
                    "status": "MISSING",
                }
                self.summary_stats["total_metrics"] += 1
                self.summary_stats["missing_count"] += 1
        
        self.metrics_by_chapter[chapter] = results
    
    def generate_markdown(self, quick: bool = False) -> str:
        """
        Generate markdown report.
        
        Args:
            quick: If True, generate condensed summary only
        
        Returns:
            Markdown formatted report
        """
        lines = []
        
        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"# Performance Analysis Report")
        lines.append(f"")
        lines.append(f"**Generated:** {timestamp}")
        lines.append(f"")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        stats = self.summary_stats
        lines.append(f"- **Total metrics analyzed:** {stats['total_metrics']}")
        lines.append(f"- **PASS:** {stats['pass_count']} ✅")
        lines.append(f"- **WARN:** {stats['warn_count']} ⚠️")
        lines.append(f"- **FAIL:** {stats['fail_count']} ❌")
        lines.append(f"- **MISSING:** {stats['missing_count']} ❓")
        lines.append("")
        
        if quick:
            # Quick mode: just show summary table
            lines.append("## Quick Summary by Chapter")
            lines.append("")
            lines.append("| Chapter | Description | Status | Metrics |")
            lines.append("|---------|-------------|--------|---------|")
            
            for chapter in sorted(self.chapters_analyzed):
                desc = get_chapter_description(chapter)
                results = self.metrics_by_chapter.get(chapter, {})
                
                # Count statuses
                statuses = [r["status"] for r in results.values()]
                pass_c = statuses.count("PASS")
                warn_c = statuses.count("WARN")
                fail_c = statuses.count("FAIL")
                missing_c = statuses.count("MISSING")
                
                # Overall chapter status
                if fail_c > 0:
                    status_icon = "❌ FAIL"
                elif warn_c > 0:
                    status_icon = "⚠️ WARN"
                elif pass_c > 0:
                    status_icon = "✅ PASS"
                else:
                    status_icon = "❓ NO DATA"
                
                metrics_summary = f"{pass_c}✅ {warn_c}⚠️ {fail_c}❌ {missing_c}❓"
                lines.append(f"| {chapter.upper()} | {desc} | {status_icon} | {metrics_summary} |")
            
            lines.append("")
        else:
            # Full mode: detailed breakdown per chapter
            lines.append("## Detailed Results by Chapter")
            lines.append("")
            
            for chapter in sorted(self.chapters_analyzed):
                desc = get_chapter_description(chapter)
                results = self.metrics_by_chapter.get(chapter, {})
                
                if not results:
                    continue
                
                lines.append(f"### {chapter.upper()}: {desc}")
                lines.append("")
                
                # Count statuses for chapter summary
                statuses = [r["status"] for r in results.values()]
                fail_c = statuses.count("FAIL")
                warn_c = statuses.count("WARN")
                pass_c = statuses.count("PASS")
                
                if fail_c > 0:
                    lines.append("**Overall Status:** ❌ FAIL")
                elif warn_c > 0:
                    lines.append("**Overall Status:** ⚠️ WARN")
                else:
                    lines.append("**Overall Status:** ✅ PASS")
                lines.append("")
                
                # Metrics table
                lines.append("| Metric | Target | Actual | Status |")
                lines.append("|--------|--------|--------|--------|")
                
                for metric_name, result in sorted(results.items()):
                    target_def = result["target"]
                    actual = result["actual"]
                    status = result["status"]
                    
                    # Format target
                    target_str = format_value(target_def["target"], target_def["unit"])
                    
                    # Format actual
                    if actual is not None:
                        actual_str = format_value(actual, target_def["unit"])
                    else:
                        actual_str = "N/A"
                    
                    # Status icon
                    status_icon = {
                        "PASS": "✅ PASS",
                        "WARN": "⚠️ WARN",
                        "FAIL": "❌ FAIL",
                        "MISSING": "❓ MISSING",
                    }.get(status, status)
                    
                    # Format metric name
                    metric_display = metric_name.replace("_", " ").title()
                    
                    lines.append(f"| {metric_display} | {target_str} | {actual_str} | {status_icon} |")
                
                lines.append("")
        
        # Recommendations
        if not quick and (stats["fail_count"] > 0 or stats["warn_count"] > 0):
            lines.append("## Recommendations")
            lines.append("")
            
            for chapter in sorted(self.chapters_analyzed):
                results = self.metrics_by_chapter.get(chapter, {})
                
                failed = [m for m, r in results.items() if r["status"] == "FAIL"]
                warned = [m for m, r in results.items() if r["status"] == "WARN"]
                
                if failed or warned:
                    desc = get_chapter_description(chapter)
                    lines.append(f"### {chapter.upper()}: {desc}")
                    lines.append("")
                    
                    if failed:
                        lines.append("**Failed metrics:**")
                        for metric in failed:
                            lines.append(f"- `{metric}`: Below minimum threshold")
                        lines.append("")
                    
                    if warned:
                        lines.append("**Warning metrics:**")
                        for metric in warned:
                            lines.append(f"- `{metric}`: Below target but above minimum")
                        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by automated performance analysis system*")
        lines.append("")
        
        return "\n".join(lines)
    
    def write_report(self, output_path: Path, quick: bool = False):
        """
        Write report to file.
        
        Args:
            output_path: Path to output markdown file
            quick: If True, generate condensed report
        """
        content = self.generate_markdown(quick=quick)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)


def generate_report_from_metrics(
    metrics: Dict[str, float],
    output_path: Optional[Path] = None,
    quick: bool = False
) -> str:
    """
    High-level function to generate a report from flat metrics dictionary.
    
    Args:
        metrics: Dictionary of metric_name -> value
        output_path: Optional path to write report
        quick: If True, generate condensed report
    
    Returns:
        Markdown formatted report
    """
    report = PerformanceReport()
    
    # Group metrics by chapter
    metrics_by_chapter: Dict[str, Dict[str, float]] = {}
    
    for metric_name, value in metrics.items():
        original_name = metric_name
        chapter = None
        
        # Strip common prefixes
        for prefix in ["benchmark_", "test_outputs_", "ncu_", "nsys_", "pytorch_"]:
            if metric_name.startswith(prefix):
                metric_name = metric_name[len(prefix):]
                break
        
        # Try to extract chapter from metric name
        if metric_name.startswith("ch") and "_" in metric_name:
            parts = metric_name.split("_", 1)
            if parts[0] in [f"ch{i}" for i in range(1, 21)]:
                chapter = parts[0]
                metric_name = parts[1]
        
        # Try exact match first
        if chapter:
            ch_metrics = get_chapter_metrics(chapter)
            if metric_name not in ch_metrics:
                # Try fuzzy matching within the chapter
                metric_normalized = metric_name.replace("_", "").lower()
                matched = False
                for target_metric in ch_metrics.keys():
                    target_normalized = target_metric.replace("_", "").lower()
                    if metric_normalized in target_normalized or target_normalized in metric_normalized:
                        metric_name = target_metric  # Use the target metric name
                        matched = True
                        break
                # If still no match, keep the original metric name (won't be reported but won't cause errors)
        elif chapter is None:
            # Try to match against known chapter metrics
            for ch in get_all_chapters():
                ch_metrics = get_chapter_metrics(ch)
                if metric_name in ch_metrics:
                    chapter = ch
                    break
                # Try fuzzy matching
                metric_normalized = metric_name.replace("_", "").lower()
                for target_metric in ch_metrics.keys():
                    target_normalized = target_metric.replace("_", "").lower()
                    if metric_normalized in target_normalized or target_normalized in metric_normalized:
                        chapter = ch
                        metric_name = target_metric  # Use the target metric name
                        break
                if chapter:
                    break
        
        # Check if it's an overall metric
        if chapter is None and "overall" in TARGETS:
            if metric_name in TARGETS["overall"]:
                chapter = "overall"
        
        if chapter:
            if chapter not in metrics_by_chapter:
                metrics_by_chapter[chapter] = {}
            # Only add if it's a known metric
            ch_metrics = get_chapter_metrics(chapter)
            if metric_name in ch_metrics:
                metrics_by_chapter[chapter][metric_name] = value
    
    # Add metrics to report
    for chapter, ch_metrics in metrics_by_chapter.items():
        report.add_chapter_metrics(chapter, ch_metrics)
    
    # Generate and optionally write report
    markdown = report.generate_markdown(quick=quick)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
    
    return markdown

