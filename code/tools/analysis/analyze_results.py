#!/usr/bin/env python3
"""
Main analysis orchestrator for performance results.
Discovers profiler outputs, extracts metrics, validates against targets,
and generates markdown reports.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Add tools directory to path for imports
TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))

from metric_extractor import discover_and_extract_all, flatten_metrics
from report_generator import generate_report_from_metrics


def find_latest_results_dir(code_root: Path, pattern: str = "test_results_*") -> Optional[Path]:
    """
    Find the most recent results directory.
    
    Args:
        code_root: Root directory to search
        pattern: Glob pattern for directories
    
    Returns:
        Path to latest directory or None
    """
    dirs = list(code_root.glob(pattern))
    if not dirs:
        return None
    
    # Sort by modification time
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def discover_all_result_directories(code_root: Path) -> List[Path]:
    """
    Discover all result directories.
    
    Args:
        code_root: Root directory to search
    
    Returns:
        List of result directory paths
    """
    patterns = [
        "test_results_*",
        "profiles_*",
        "profile_runs/harness/*",
    ]
    
    all_dirs = []
    for pattern in patterns:
        dirs = list(code_root.glob(pattern))
        all_dirs.extend([d for d in dirs if d.is_dir()])
    
    return all_dirs


def analyze_directory(
    directory: Path,
    output_path: Optional[Path] = None,
    quick: bool = False,
    verbose: bool = False
) -> str:
    """
    Analyze a single results directory.
    
    Args:
        directory: Path to results directory
        output_path: Optional path to write report
        quick: If True, generate quick summary only
        verbose: If True, print detailed extraction info
    
    Returns:
        Generated markdown report
    """
    if verbose:
        print(f"Analyzing directory: {directory}")
        print("=" * 80)
    
    # Extract metrics from all sources
    if verbose:
        print("Discovering profiler outputs...")
    
    nested_results = discover_and_extract_all(directory)
    
    if verbose:
        print(f"Found:")
        print(f"  - Test outputs: {len(nested_results['test_outputs'])}")
        print(f"  - Benchmark files: {len(nested_results['benchmark'])}")
        print(f"  - NCU reports: {len(nested_results['ncu'])}")
        print(f"  - Nsys reports: {len(nested_results['nsys'])}")
        print(f"  - PyTorch profiles: {len(nested_results['pytorch'])}")
        print()
    
    # Flatten metrics
    if verbose:
        print("Extracting metrics...")
    
    metrics = flatten_metrics(nested_results)
    
    if verbose:
        print(f"Extracted {len(metrics)} metrics")
        print()
    
    # Generate report
    if verbose:
        print("Generating report...")
    
    report = generate_report_from_metrics(
        metrics,
        output_path=output_path,
        quick=quick
    )
    
    if verbose and output_path:
        print(f"Report written to: {output_path}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze performance test results and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest test results
  python tools/analyze_results.py
  
  # Analyze specific directory
  python tools/analyze_results.py --input test_results_20251028_063307
  
  # Generate quick summary
  python tools/analyze_results.py --quick
  
  # Analyze all historical results
  python tools/analyze_results.py --all
  
  # Write to specific output file
  python tools/analyze_results.py --output docs/analysis_latest.md
  
  # Analyze power efficiency
  python tools/analyze_results.py --power-file power.json --throughput-file results.json
        """
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory to analyze (default: latest test_results_*)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output markdown file (default: stdout or input_dir/analysis.md)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate quick summary only (faster)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all historical result directories"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed extraction information"
    )
    
    parser.add_argument(
        "--power-file",
        type=Path,
        help="Power monitoring JSON file for efficiency analysis"
    )
    
    parser.add_argument(
        "--throughput-file",
        type=Path,
        help="Throughput JSON file for efficiency analysis"
    )
    
    args = parser.parse_args()
    
    # Determine code root
    code_root = Path(__file__).resolve().parents[1]
    
    # Handle power efficiency analysis if requested
    if args.power_file and args.throughput_file:
        try:
            # Import power efficiency analyzer
            import subprocess
            result = subprocess.run([
                sys.executable,
                str(code_root / "tools" / "power_efficiency_analyzer.py"),
                "--power-file", str(args.power_file),
                "--throughput-file", str(args.throughput_file),
                "--output", str(args.output) if args.output else "-",
            ], check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Power efficiency analysis failed: {e}", file=sys.stderr)
            return 1
    
    if args.all:
        # Analyze all result directories
        directories = discover_all_result_directories(code_root)
        
        if not directories:
            print("No result directories found", file=sys.stderr)
            return 1
        
        print(f"Found {len(directories)} result directories to analyze")
        print()
        
        for directory in directories:
            print(f"Analyzing: {directory.name}")
            output_path = directory / "analysis.md"
            
            try:
                analyze_directory(
                    directory,
                    output_path=output_path,
                    quick=args.quick,
                    verbose=args.verbose
                )
                print(f"  ✓ Report written to: {output_path}")
            except Exception as e:
                print(f"  ✗ Error: {e}", file=sys.stderr)
            
            print()
        
        return 0
    
    # Analyze single directory
    if args.input:
        directory = args.input
        if not directory.is_absolute():
            directory = code_root / directory
    else:
        # Find latest test results
        directory = find_latest_results_dir(code_root)
        if not directory:
            print("No test results found. Use --input to specify a directory.", file=sys.stderr)
            return 1
    
    if not directory.exists():
        print(f"Directory not found: {directory}", file=sys.stderr)
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = code_root / output_path
    else:
        if args.quick or sys.stdout.isatty():
            # Quick mode or interactive: write to input directory
            output_path = directory / "analysis.md"
        else:
            # Non-interactive: stdout only
            output_path = None
    
    # Run analysis
    try:
        report = analyze_directory(
            directory,
            output_path=output_path,
            quick=args.quick,
            verbose=args.verbose
        )
        
        # Print to stdout if no output file or if verbose
        if output_path is None or args.verbose:
            print()
            print(report)
        
        return 0
    
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


