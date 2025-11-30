#!/usr/bin/env python3
"""Linter to check that all benchmarks follow the benchmark contract.

Usage:
    python core/scripts/linting/check_benchmarks.py                    # Check all benchmarks
    python core/scripts/linting/check_benchmarks.py ch01/               # Check specific chapter
    python core/scripts/linting/check_benchmarks.py --fix              # Auto-fix issues (if possible)
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from core.benchmark.contract import check_benchmark_file
from core.discovery import discover_benchmark_pairs, discover_all_chapters


def main():
    parser = argparse.ArgumentParser(description="Check benchmarks follow the contract")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to check (default: all chapters)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues (not implemented yet)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--run-setup",
        action="store_true",
        help="Actually import and instantiate benchmarks during validation (WARNING: executes module code and constructors, requires CUDA for BaseBenchmark, not suitable for pre-commit hooks). By default, uses AST parsing for side-effect free validation.",
    )
    args = parser.parse_args()
    
    if args.fix:
        print("Auto-fix is not yet implemented")
        sys.exit(1)
    
    # Find benchmark files
    if args.paths:
        benchmark_files = []
        for path_str in args.paths:
            path = Path(path_str).resolve()
            if path.is_file() and path.suffix == ".py":
                benchmark_files.append(path)
            elif path.is_dir():
                # Find all benchmark files in directory
                for file in path.rglob("baseline_*.py"):
                    benchmark_files.append(file)
                for file in path.rglob("optimized_*.py"):
                    benchmark_files.append(file)
    else:
        # Check all benchmarks using discover_benchmark_pairs
        benchmark_files = []
        pairs = discover_benchmark_pairs(repo_root, chapter="all")
        for baseline, optimized_list, _ in pairs:
            if baseline:
                benchmark_files.append(baseline)
            # optimized_list is a list of Path objects, not a single Path
            for optimized_path in optimized_list:
                benchmark_files.append(optimized_path)
    
    if not benchmark_files:
        print("No benchmark files found")
        return 0
    
    print(f"Checking {len(benchmark_files)} benchmark files...")
    print()
    
    total_errors = 0
    total_warnings = 0
    failed_files = []
    
    for file_path in sorted(benchmark_files):
        is_valid, errors, warnings = check_benchmark_file(file_path, run_setup=args.run_setup)
        
        if errors or warnings:
            print(f"❌ {file_path.relative_to(repo_root)}")
            if errors:
                total_errors += len(errors)
                for error in errors:
                    print(f"   ERROR: {error}")
            if warnings:
                total_warnings += len(warnings)
                if args.verbose:
                    for warning in warnings:
                        print(f"   WARNING: {warning}")
            failed_files.append(file_path)
        else:
            if args.verbose:
                print(f"✓ {file_path.relative_to(repo_root)}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files checked: {len(benchmark_files)}")
    print(f"Files with errors: {len(failed_files)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    
    if failed_files:
        print()
        print("Failed files:")
        for file_path in failed_files:
            print(f"  - {file_path.relative_to(repo_root)}")
    
    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

