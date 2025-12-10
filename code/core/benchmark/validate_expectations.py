#!/usr/bin/env python3
"""Validation CLI tool for benchmark expectation files.

This tool scans expectation files and reports data integrity issues:
- Speedup mismatches (stored speedup differs from computed ratio)
- Masked regressions (speedup=1.0 hiding actual regression)
- Metadata drift (metadata speedup differs from metrics)
- Missing provenance (required fields missing)

Usage:
    python -m core.benchmark.validate_expectations [paths...] [--fix] [--verbose] [--strict]

Examples:
    # Validate all expectation files in repo
    python -m core.benchmark.validate_expectations

    # Validate specific chapter
    python -m core.benchmark.validate_expectations ch07/

    # Fix issues (recompute speedups from timing)
    python -m core.benchmark.validate_expectations --fix

    # Strict mode for CI (exit 1 on issues)
    python -m core.benchmark.validate_expectations --strict
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.benchmark.expectations import (
    EXPECTATION_FILENAME_TEMPLATE,
    SCHEMA_VERSION,
    SPEEDUP_TOLERANCE,
    ExpectationsStore,
    ValidationIssue,
    ValidationReport,
    compute_speedup,
)


def find_expectation_files(paths: List[Path], repo_root: Path) -> List[Path]:
    """Find all expectation files in the given paths.

    Args:
        paths: List of paths (files or directories) to search
        repo_root: Repository root directory

    Returns:
        List of expectation file paths
    """
    files: List[Path] = []

    if not paths:
        # Default: search entire repo
        paths = [repo_root]

    for path in paths:
        if path.is_file():
            if path.name.startswith("expectations_") and path.suffix == ".json":
                files.append(path)
        elif path.is_dir():
            # Recursively find expectation files
            for exp_file in path.rglob("expectations_*.json"):
                files.append(exp_file)

    return sorted(set(files))


def validate_file(
    path: Path,
    verbose: bool = False,
) -> ValidationReport:
    """Validate a single expectation file.

    Args:
        path: Path to expectation file
        verbose: Print detailed output

    Returns:
        ValidationReport with issues found
    """
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return ValidationReport(
            issues=[
                ValidationIssue(
                    example_key="<file>",
                    issue_type="json_error",
                    message=f"Failed to parse JSON: {e}",
                    stored_value=None,
                    expected_value=None,
                )
            ],
            total_entries=0,
            valid_entries=0,
        )

    hardware_key = data.get("hardware_key", "unknown")
    chapter_dir = path.parent

    # Create store to use validation methods
    store = ExpectationsStore(chapter_dir, hardware_key)
    store._data = data  # Use loaded data directly

    return store.validate_all()


def fix_file(
    path: Path,
    verbose: bool = False,
    backup: bool = True,
) -> bool:
    """Fix issues in an expectation file by recomputing speedups.

    Args:
        path: Path to expectation file
        verbose: Print detailed output
        backup: Create backup before modifying

    Returns:
        True if file was modified
    """
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  ERROR: Cannot fix {path} - JSON parse error")
        return False

    modified = False
    examples = data.get("examples", {})

    for example_key, entry in examples.items():
        metrics = entry.get("metrics", {})
        metadata = entry.get("metadata", {})

        baseline_time = metrics.get("baseline_time_ms", 0.0)
        optimized_time = metrics.get("best_optimized_time_ms", 0.0)

        if baseline_time > 0 and optimized_time > 0:
            computed_speedup = compute_speedup(baseline_time, optimized_time)
            stored_speedup = metrics.get("best_speedup")

            # Check if speedup needs fixing
            if stored_speedup is None or abs(stored_speedup - computed_speedup) > SPEEDUP_TOLERANCE:
                if verbose:
                    print(f"    Fixing {example_key}: {stored_speedup} -> {computed_speedup:.6f}")
                metrics["best_speedup"] = computed_speedup
                metrics["best_optimized_speedup"] = computed_speedup
                metrics["is_regression"] = computed_speedup < 1.0
                modified = True

            # Fix metadata speedup if needed
            metadata_speedup = metadata.get("best_optimization_speedup")
            if metadata_speedup is not None and abs(metadata_speedup - computed_speedup) > SPEEDUP_TOLERANCE:
                if verbose:
                    print(f"    Fixing {example_key} metadata: {metadata_speedup} -> {computed_speedup:.6f}")
                metadata["best_optimization_speedup"] = computed_speedup
                modified = True

    if modified:
        # Update schema version
        data["schema_version"] = SCHEMA_VERSION

        if backup:
            backup_path = path.with_suffix(".json.bak")
            shutil.copy2(path, backup_path)
            if verbose:
                print(f"  Created backup: {backup_path}")

        # Write fixed data
        serialized = json.dumps(data, indent=2, sort_keys=True)
        path.write_text(serialized + "\n")

    return modified


def format_issue(issue: ValidationIssue, verbose: bool = False) -> str:
    """Format a validation issue for display."""
    parts = [f"    [{issue.issue_type}] {issue.example_key}: {issue.message}"]

    if verbose:
        parts.append(f"      Stored: {issue.stored_value}")
        parts.append(f"      Expected: {issue.expected_value}")
        if issue.delta_pct is not None:
            parts.append(f"      Delta: {issue.delta_pct:.2f}%")

    return "\n".join(parts)


def validate_expectations(
    paths: Optional[List[str]] = None,
    fix: bool = False,
    verbose: bool = False,
    strict: bool = False,
) -> int:
    """Main validation function.

    Args:
        paths: Paths to validate (files or directories)
        fix: Recompute speedups to fix issues
        verbose: Print detailed output
        strict: Return exit code 1 if any issues found

    Returns:
        Exit code: 0=valid, 1=issues (strict mode), 2=error
    """
    # Find repo root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]  # core/benchmark/validate_expectations.py -> repo_root

    # Convert paths to Path objects
    path_list: List[Path] = []
    if paths:
        for p in paths:
            path = Path(p)
            if not path.is_absolute():
                path = repo_root / path
            path_list.append(path)

    # Find expectation files
    files = find_expectation_files(path_list, repo_root)

    if not files:
        print("No expectation files found.")
        return 0

    print(f"Validating {len(files)} expectation file(s)...\n")

    total_issues = 0
    total_entries = 0
    valid_entries = 0
    files_with_issues = 0
    files_fixed = 0

    for path in files:
        rel_path = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
        print(f"  {rel_path}")

        report = validate_file(path, verbose)
        total_entries += report.total_entries
        valid_entries += report.valid_entries

        if report.has_issues:
            files_with_issues += 1
            total_issues += len(report.issues)

            for issue in report.issues:
                print(format_issue(issue, verbose))

            if fix:
                if fix_file(path, verbose):
                    files_fixed += 1
                    print(f"    ✓ Fixed {len(report.issues)} issue(s)")
        else:
            print(f"    ✓ {report.total_entries} entries, all valid")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Files scanned: {len(files)}")
    print(f"  Files with issues: {files_with_issues}")
    print(f"  Total entries: {total_entries}")
    print(f"  Valid entries: {valid_entries}")
    print(f"  Total issues: {total_issues}")

    if fix:
        print(f"  Files fixed: {files_fixed}")

    if total_issues > 0:
        issue_types = {}
        for path in files:
            report = validate_file(path, False)
            for issue in report.issues:
                issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1

        print("\n  Issues by type:")
        for issue_type, count in sorted(issue_types.items()):
            print(f"    {issue_type}: {count}")

    print(f"{'='*60}")

    if total_issues > 0:
        if strict:
            print("\n❌ FAILED: Issues found in strict mode")
            return 1
        else:
            print("\n⚠️  Issues found (use --fix to repair, --strict for CI)")
            return 0
    else:
        print("\n✓ All expectation files are valid")
        return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate benchmark expectation files for data integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Validate all files
  %(prog)s ch07/               # Validate specific chapter
  %(prog)s --fix               # Fix issues by recomputing speedups
  %(prog)s --strict            # CI mode: exit 1 if issues found
  %(prog)s --verbose           # Show detailed issue information
""",
    )

    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to validate (files or directories). Default: entire repo",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Recompute speedups from timing values to fix inconsistencies",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed issue information",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if any issues are found (for CI)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files when fixing (use with --fix)",
    )

    args = parser.parse_args()

    try:
        exit_code = validate_expectations(
            paths=args.paths if args.paths else None,
            fix=args.fix,
            verbose=args.verbose,
            strict=args.strict,
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()


