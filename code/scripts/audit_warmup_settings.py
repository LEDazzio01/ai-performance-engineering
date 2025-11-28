#!/usr/bin/env python3
"""Audit script to detect insufficient warmup iterations in benchmarks.

This script scans all Python files in the codebase to find benchmarks with
warmup iterations below the minimum threshold. Low warmup can cause JIT/compile
overhead to be included in measurements, leading to inaccurate benchmark results.

Usage:
    python scripts/audit_warmup_settings.py [--fix] [--verbose] [--paths ...]

The script will:
1. Scan all ch*/ and labs/*/ directories for Python benchmark files
2. Detect warmup settings in BenchmarkConfig instances
3. Flag any warmup value below MINIMUM_WARMUP_ITERATIONS (5)
4. Optionally detect torch.compile usage and recommend higher warmup

Exit codes:
    0 - All checks passed
    1 - Issues found (insufficient warmup)
    2 - Error during execution
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set


# Import from benchmark_defaults if available, otherwise use hardcoded values
try:
    from common.python.benchmark_defaults import (
        MINIMUM_WARMUP_ITERATIONS,
        RECOMMENDED_WARMUP_TORCH_COMPILE,
        RECOMMENDED_WARMUP_TRITON,
    )
except ImportError:
    MINIMUM_WARMUP_ITERATIONS = 5
    RECOMMENDED_WARMUP_TORCH_COMPILE = 10
    RECOMMENDED_WARMUP_TRITON = 10


class WarmupIssue(NamedTuple):
    """Represents a warmup configuration issue."""
    file_path: Path
    line_number: int
    warmup_value: int
    uses_torch_compile: bool
    uses_triton: bool
    recommended_warmup: int
    code_snippet: str


def detect_torch_compile_usage(file_content: str) -> bool:
    """Detect if file uses torch.compile or related compilation."""
    patterns = [
        r'torch\.compile\s*\(',
        r'compile_fn\s*\(',
        r'compile_model\s*\(',
        r'compile_callable\s*\(',
        r'@torch\.compile',
        r'mode\s*=\s*["\']reduce-overhead["\']',
        r'mode\s*=\s*["\']max-autotune["\']',
    ]
    for pattern in patterns:
        if re.search(pattern, file_content):
            return True
    return False


def detect_triton_usage(file_content: str) -> bool:
    """Detect if file uses Triton kernels."""
    patterns = [
        r'import\s+triton',
        r'from\s+triton',
        r'@triton\.jit',
        r'triton\.testing',
        r'tl\.',  # triton language
    ]
    for pattern in patterns:
        if re.search(pattern, file_content):
            return True
    return False


def extract_warmup_from_config(file_content: str, file_path: Path) -> List[WarmupIssue]:
    """Extract warmup values from BenchmarkConfig instances in file."""
    issues: List[WarmupIssue] = []
    
    uses_torch_compile = detect_torch_compile_usage(file_content)
    uses_triton = detect_triton_usage(file_content)
    
    # Determine recommended warmup based on features used
    if uses_torch_compile:
        recommended = RECOMMENDED_WARMUP_TORCH_COMPILE
    elif uses_triton:
        recommended = RECOMMENDED_WARMUP_TRITON
    else:
        recommended = MINIMUM_WARMUP_ITERATIONS
    
    # Pattern to find warmup= in BenchmarkConfig or similar
    # NOTE: We exclude torch.profiler.schedule(warmup=...) which is different
    patterns = [
        # BenchmarkConfig(warmup=X, ...)
        r'BenchmarkConfig\s*\([^)]*warmup\s*=\s*(\d+)',
        # warmup=X in __init__ super().__init__(warmup=X, ...) for CudaBinaryBenchmark
        r'super\(\)\.__init__\s*\([^)]*\bwarmup\s*=\s*(\d+)',
    ]
    
    # Patterns to EXCLUDE (false positives)
    exclude_patterns = [
        r'profiler\.schedule\s*\([^)]*warmup',  # torch.profiler.schedule
        r'do_bench\s*\([^)]*warmup\s*=\s*0',  # triton do_bench with warmup=0 (harness handles)
        r'# We handle warmup',  # Explicit comment indicating handled elsewhere
        r'num_warmup',  # Different variable name (function parameter)
        r'gen_mkl_autotuner',  # Third-party function
    ]
    
    # Check if line should be excluded
    def should_exclude_line(line: str) -> bool:
        for pattern in exclude_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    lines = file_content.split('\n')
    
    for pattern in patterns:
        for match in re.finditer(pattern, file_content):
            warmup_value = int(match.group(1))
            
            # Find line number
            char_pos = match.start()
            line_num = file_content[:char_pos].count('\n') + 1
            
            # Get code snippet (the line containing the warmup setting)
            if 0 < line_num <= len(lines):
                snippet = lines[line_num - 1].strip()
            else:
                snippet = match.group(0)
            
            # Check if this line should be excluded (false positive)
            if should_exclude_line(snippet):
                continue
            
            # Also check the full match context
            context_start = max(0, char_pos - 50)
            context_end = min(len(file_content), char_pos + 100)
            context = file_content[context_start:context_end]
            if should_exclude_line(context):
                continue
            
            # Only flag if below minimum
            if warmup_value < MINIMUM_WARMUP_ITERATIONS:
                issues.append(WarmupIssue(
                    file_path=file_path,
                    line_number=line_num,
                    warmup_value=warmup_value,
                    uses_torch_compile=uses_torch_compile,
                    uses_triton=uses_triton,
                    recommended_warmup=recommended,
                    code_snippet=snippet,
                ))
    
    return issues


def scan_directory(directory: Path, verbose: bool = False) -> List[WarmupIssue]:
    """Scan a directory for warmup issues."""
    issues: List[WarmupIssue] = []
    
    # Skip certain directories (including virtual environments and third-party code)
    skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', 
                 'node_modules', '.tox', '.eggs', '*.egg-info', '.venv', 'venv',
                 'site-packages', 'vendor', 'third_party'}
    
    for root, dirs, files in os.walk(directory):
        # Filter out skip directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith('.egg-info')]
        
        for filename in files:
            if not filename.endswith('.py'):
                continue
            
            # Skip test files and non-benchmark files
            if filename.startswith('test_') or filename == 'conftest.py':
                continue
            
            file_path = Path(root) / filename
            
            try:
                content = file_path.read_text(encoding='utf-8')
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not read {file_path}: {e}", file=sys.stderr)
                continue
            
            # Only check files that look like benchmarks
            if 'BenchmarkConfig' not in content and 'warmup' not in content:
                continue
            
            file_issues = extract_warmup_from_config(content, file_path)
            issues.extend(file_issues)
            
            if verbose and file_issues:
                print(f"  Found {len(file_issues)} issue(s) in {file_path}")
    
    return issues


def format_issue(issue: WarmupIssue, repo_root: Path) -> str:
    """Format an issue for display."""
    rel_path = issue.file_path.relative_to(repo_root) if issue.file_path.is_relative_to(repo_root) else issue.file_path
    
    features = []
    if issue.uses_torch_compile:
        features.append("torch.compile")
    if issue.uses_triton:
        features.append("Triton")
    features_str = f" [uses: {', '.join(features)}]" if features else ""
    
    return (
        f"  {rel_path}:{issue.line_number}\n"
        f"    Current warmup: {issue.warmup_value} (minimum: {MINIMUM_WARMUP_ITERATIONS}, recommended: {issue.recommended_warmup}){features_str}\n"
        f"    Code: {issue.code_snippet[:80]}{'...' if len(issue.code_snippet) > 80 else ''}"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Audit benchmark warmup settings across the codebase.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress information'
    )
    parser.add_argument(
        '--paths',
        nargs='*',
        help='Specific paths to scan (default: ch*/, labs/)'
    )
    parser.add_argument(
        '--check-recommended',
        action='store_true',
        help='Also flag warmup below recommended (not just minimum)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Determine repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    if not (repo_root / 'common' / 'python').exists():
        print("Error: Could not find repo root. Run from scripts/ directory.", file=sys.stderr)
        sys.exit(2)
    
    # Determine paths to scan
    if args.paths:
        scan_paths = [Path(p) for p in args.paths]
    else:
        # Default: scan ch*/ and labs/*/
        scan_paths = []
        for pattern in ['ch*', 'labs/*']:
            scan_paths.extend(repo_root.glob(pattern))
        # Also scan common/python for any benchmark definitions there
        scan_paths.append(repo_root / 'common' / 'python')
    
    print("=" * 70)
    print("BENCHMARK WARMUP AUDIT")
    print("=" * 70)
    print(f"Minimum warmup required: {MINIMUM_WARMUP_ITERATIONS}")
    print(f"Recommended for torch.compile: {RECOMMENDED_WARMUP_TORCH_COMPILE}")
    print(f"Recommended for Triton: {RECOMMENDED_WARMUP_TRITON}")
    print()
    
    all_issues: List[WarmupIssue] = []
    
    for scan_path in sorted(set(scan_paths)):
        if not scan_path.exists():
            if args.verbose:
                print(f"Skipping non-existent path: {scan_path}")
            continue
        
        if args.verbose:
            print(f"Scanning: {scan_path}")
        
        issues = scan_directory(scan_path, verbose=args.verbose)
        all_issues.extend(issues)
    
    # Filter issues based on --check-recommended flag
    if not args.check_recommended:
        # Only keep issues where warmup < MINIMUM
        all_issues = [i for i in all_issues if i.warmup_value < MINIMUM_WARMUP_ITERATIONS]
    
    # Deduplicate issues (same file, same line)
    seen: Set[tuple] = set()
    unique_issues: List[WarmupIssue] = []
    for issue in all_issues:
        key = (issue.file_path, issue.line_number)
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)
    
    if args.json:
        import json
        output = {
            "minimum_warmup": MINIMUM_WARMUP_ITERATIONS,
            "total_scanned": len(scan_paths),
            "passing": len(scan_paths) - len(unique_issues),
            "issues": [
                {
                    "file": str(i.file_path.relative_to(repo_root) if i.file_path.is_relative_to(repo_root) else i.file_path),
                    "line": i.line_number,
                    "current": i.warmup_value,
                    "required": MINIMUM_WARMUP_ITERATIONS,
                    "recommended": i.recommended_warmup,
                    "reason": f"torch.compile requires {RECOMMENDED_WARMUP_TORCH_COMPILE}+ warmup" if i.uses_torch_compile else (
                        f"Triton requires {RECOMMENDED_WARMUP_TRITON}+ warmup" if i.uses_triton else
                        f"Minimum warmup is {MINIMUM_WARMUP_ITERATIONS}"
                    ),
                    "uses_torch_compile": i.uses_torch_compile,
                    "uses_triton": i.uses_triton,
                }
                for i in unique_issues
            ]
        }
        print(json.dumps(output, indent=2))
        sys.exit(1 if unique_issues else 0)
    
    print()
    if unique_issues:
        print(f"FOUND {len(unique_issues)} ISSUE(S):")
        print("-" * 70)
        
        # Group by file
        by_file: Dict[Path, List[WarmupIssue]] = {}
        for issue in unique_issues:
            by_file.setdefault(issue.file_path, []).append(issue)
        
        for file_path in sorted(by_file.keys()):
            for issue in by_file[file_path]:
                print(format_issue(issue, repo_root))
                print()
        
        print("-" * 70)
        print(f"Total: {len(unique_issues)} warmup issue(s) found")
        print()
        print("To fix: Update warmup values to at least {MINIMUM_WARMUP_ITERATIONS}")
        print("        For torch.compile benchmarks, use warmup={RECOMMENDED_WARMUP_TORCH_COMPILE}")
        print()
        print("WHY THIS MATTERS:")
        print("  Low warmup causes JIT/compile overhead to be INCLUDED in measurements.")
        print("  This leads to inaccurate speedup calculations and misleading results.")
        print("  torch.compile typically needs 1-3 calls to fully compile code.")
        print()
        sys.exit(1)
    else:
        print("✓ All benchmarks have sufficient warmup iterations!")
        print()
        sys.exit(0)


if __name__ == '__main__':
    main()

