#!/usr/bin/env python3
"""Validate that baseline/optimized benchmark pairs have matching signatures.

This script discovers all baseline_*.py and optimized_*.py pairs and verifies
that their input signatures match, ensuring fair performance comparisons.

Usage:
    # Validate all pairs
    python -m core.scripts.validate_benchmark_pairs
    
    # Validate specific chapter
    python -m core.scripts.validate_benchmark_pairs --chapter ch07
    
    # Generate JSON report
    python -m core.scripts.validate_benchmark_pairs --report artifacts/pair_validation.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SignatureMismatch:
    """Details of a signature mismatch between baseline and optimized."""
    key: str
    baseline_value: Any
    optimized_value: Any


@dataclass
class PairValidationResult:
    """Result of validating a single benchmark pair."""
    chapter: str
    example_name: str
    baseline_path: str
    optimized_path: str
    
    # Status
    valid: bool = False
    error: Optional[str] = None
    
    # Signature comparison
    baseline_has_signature: bool = False
    optimized_has_signature: bool = False
    signatures_match: bool = False
    mismatches: List[SignatureMismatch] = field(default_factory=list)
    
    # Extra keys
    baseline_only_keys: Set[str] = field(default_factory=set)
    optimized_only_keys: Set[str] = field(default_factory=set)


@dataclass
class ValidationReport:
    """Complete pair validation report."""
    timestamp: str
    total_pairs: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    missing_signature_pairs: int = 0
    signature_mismatch_pairs: int = 0
    error_pairs: int = 0
    results: List[PairValidationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_pairs": self.total_pairs,
                "valid_pairs": self.valid_pairs,
                "invalid_pairs": self.invalid_pairs,
                "missing_signature_pairs": self.missing_signature_pairs,
                "signature_mismatch_pairs": self.signature_mismatch_pairs,
                "error_pairs": self.error_pairs,
            },
            "results": [
                {
                    "chapter": r.chapter,
                    "example_name": r.example_name,
                    "baseline_path": r.baseline_path,
                    "optimized_path": r.optimized_path,
                    "valid": r.valid,
                    "error": r.error,
                    "baseline_has_signature": r.baseline_has_signature,
                    "optimized_has_signature": r.optimized_has_signature,
                    "signatures_match": r.signatures_match,
                    "mismatches": [
                        {"key": m.key, "baseline": m.baseline_value, "optimized": m.optimized_value}
                        for m in r.mismatches
                    ],
                    "baseline_only_keys": list(r.baseline_only_keys),
                    "optimized_only_keys": list(r.optimized_only_keys),
                }
                for r in self.results
            ],
        }


# =============================================================================
# Benchmark Loading
# =============================================================================

def load_benchmark_class(file_path: Path) -> Optional[Any]:
    """Dynamically load and instantiate a benchmark from a file."""
    try:
        spec = importlib.util.spec_from_file_location("benchmark_module", file_path)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = module
        spec.loader.exec_module(module)
        
        # Look for get_benchmark factory function
        if hasattr(module, "get_benchmark"):
            return module.get_benchmark()
        
        # Look for Benchmark class
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, "benchmark_fn") and 
                name != "BaseBenchmark"):
                return obj()
        
        return None
    except Exception as e:
        # Import errors are common due to missing dependencies
        return None
    finally:
        if "benchmark_module" in sys.modules:
            del sys.modules["benchmark_module"]


def get_input_signature_safe(benchmark: Any) -> Tuple[Optional[Dict], Optional[str]]:
    """Safely get input signature from a benchmark instance."""
    if not hasattr(benchmark, "get_input_signature"):
        return None, "Method not implemented"
    
    try:
        sig = benchmark.get_input_signature()
        if sig is None:
            return None, "Method returned None"
        return sig, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# Pair Discovery
# =============================================================================

def discover_benchmark_pairs(root_dir: Path, chapter: Optional[str] = None) -> Dict[str, Dict[str, Path]]:
    """Discover baseline/optimized benchmark pairs.
    
    Returns:
        Dict mapping (chapter, example_name) to {baseline: path, optimized: path}
    """
    pairs: Dict[Tuple[str, str], Dict[str, Path]] = defaultdict(dict)
    
    # Determine search directories
    if chapter:
        if chapter.startswith("labs/"):
            search_dirs = [root_dir / chapter]
        else:
            search_dirs = [root_dir / chapter]
    else:
        search_dirs = [
            d for d in root_dir.iterdir()
            if d.is_dir() and (d.name.startswith("ch") or d.name == "labs")
        ]
        labs_dir = root_dir / "labs"
        if labs_dir.exists():
            search_dirs.extend(d for d in labs_dir.iterdir() if d.is_dir())
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Get chapter name
        relative = search_dir.relative_to(root_dir)
        chapter_name = str(relative)
        
        # Find all baseline and optimized files
        baseline_files = list(search_dir.glob("baseline_*.py"))
        optimized_files = list(search_dir.glob("optimized_*.py"))
        
        # Also check subdirectories
        baseline_files.extend(search_dir.glob("**/baseline_*.py"))
        optimized_files.extend(search_dir.glob("**/optimized_*.py"))
        
        # Extract example names and build pairs
        baseline_by_name = {}
        for f in baseline_files:
            # Example: baseline_attention.py -> attention
            name = f.stem.replace("baseline_", "")
            baseline_by_name[name] = f
        
        optimized_by_name = {}
        for f in optimized_files:
            # Example: optimized_attention.py -> attention
            name = f.stem.replace("optimized_", "")
            optimized_by_name[name] = f
        
        # Match pairs
        all_names = set(baseline_by_name.keys()) | set(optimized_by_name.keys())
        for name in all_names:
            if name in baseline_by_name:
                pairs[(chapter_name, name)]["baseline"] = baseline_by_name[name]
            if name in optimized_by_name:
                pairs[(chapter_name, name)]["optimized"] = optimized_by_name[name]
    
    return {f"{ch}:{name}": paths for (ch, name), paths in pairs.items()}


# =============================================================================
# Signature Comparison
# =============================================================================

def compare_signatures(
    baseline_sig: Dict[str, Any],
    optimized_sig: Dict[str, Any],
) -> Tuple[bool, List[SignatureMismatch], Set[str], Set[str]]:
    """Compare two input signatures.
    
    Returns:
        Tuple of (match, mismatches, baseline_only_keys, optimized_only_keys)
    """
    mismatches: List[SignatureMismatch] = []
    
    baseline_keys = set(baseline_sig.keys())
    optimized_keys = set(optimized_sig.keys())
    
    baseline_only = baseline_keys - optimized_keys
    optimized_only = optimized_keys - baseline_keys
    common_keys = baseline_keys & optimized_keys
    
    # Compare common keys
    for key in common_keys:
        baseline_val = baseline_sig[key]
        optimized_val = optimized_sig[key]
        
        if baseline_val != optimized_val:
            mismatches.append(SignatureMismatch(
                key=key,
                baseline_value=baseline_val,
                optimized_value=optimized_val,
            ))
    
    match = len(mismatches) == 0 and len(baseline_only) == 0 and len(optimized_only) == 0
    return match, mismatches, baseline_only, optimized_only


# =============================================================================
# Validation
# =============================================================================

def validate_pair(
    chapter: str,
    example_name: str,
    baseline_path: Path,
    optimized_path: Path,
) -> PairValidationResult:
    """Validate a single baseline/optimized pair."""
    result = PairValidationResult(
        chapter=chapter,
        example_name=example_name,
        baseline_path=str(baseline_path),
        optimized_path=str(optimized_path),
    )
    
    # Load benchmarks
    baseline_benchmark = load_benchmark_class(baseline_path)
    if baseline_benchmark is None:
        result.error = f"Failed to load baseline benchmark"
        return result
    
    optimized_benchmark = load_benchmark_class(optimized_path)
    if optimized_benchmark is None:
        result.error = f"Failed to load optimized benchmark"
        return result
    
    # Get signatures
    baseline_sig, baseline_err = get_input_signature_safe(baseline_benchmark)
    optimized_sig, optimized_err = get_input_signature_safe(optimized_benchmark)
    
    result.baseline_has_signature = baseline_sig is not None
    result.optimized_has_signature = optimized_sig is not None
    
    if baseline_sig is None and optimized_sig is None:
        result.error = f"Neither benchmark has get_input_signature"
        return result
    
    if baseline_sig is None:
        result.error = f"Baseline missing signature: {baseline_err}"
        return result
    
    if optimized_sig is None:
        result.error = f"Optimized missing signature: {optimized_err}"
        return result
    
    # Compare signatures
    match, mismatches, baseline_only, optimized_only = compare_signatures(
        baseline_sig, optimized_sig
    )
    
    result.signatures_match = match
    result.mismatches = mismatches
    result.baseline_only_keys = baseline_only
    result.optimized_only_keys = optimized_only
    result.valid = match
    
    return result


def validate_all_pairs(
    root_dir: Path,
    chapter: Optional[str] = None,
) -> ValidationReport:
    """Validate all benchmark pairs."""
    report = ValidationReport(timestamp=datetime.now().isoformat())
    
    # Discover pairs
    pairs = discover_benchmark_pairs(root_dir, chapter)
    report.total_pairs = len(pairs)
    
    print(f"Found {len(pairs)} benchmark pairs to validate")
    print()
    
    for pair_name, paths in sorted(pairs.items()):
        chapter_name, example_name = pair_name.split(":", 1)
        
        # Check if we have both baseline and optimized
        if "baseline" not in paths:
            result = PairValidationResult(
                chapter=chapter_name,
                example_name=example_name,
                baseline_path="",
                optimized_path=str(paths.get("optimized", "")),
                error="Missing baseline file",
            )
            report.results.append(result)
            report.error_pairs += 1
            continue
        
        if "optimized" not in paths:
            result = PairValidationResult(
                chapter=chapter_name,
                example_name=example_name,
                baseline_path=str(paths.get("baseline", "")),
                optimized_path="",
                error="Missing optimized file",
            )
            report.results.append(result)
            report.error_pairs += 1
            continue
        
        # Validate the pair
        result = validate_pair(
            chapter_name,
            example_name,
            paths["baseline"],
            paths["optimized"],
        )
        report.results.append(result)
        
        if result.valid:
            report.valid_pairs += 1
            print(f"  ✓ {pair_name}")
        elif result.error:
            if "missing signature" in result.error.lower() or "not implemented" in result.error.lower():
                report.missing_signature_pairs += 1
                print(f"  ○ {pair_name} - {result.error}")
            else:
                report.error_pairs += 1
                print(f"  ✗ {pair_name} - {result.error}")
        else:
            report.signature_mismatch_pairs += 1
            report.invalid_pairs += 1
            print(f"  ✗ {pair_name} - signature mismatch:")
            for m in result.mismatches:
                print(f"      {m.key}: baseline={m.baseline_value}, optimized={m.optimized_value}")
            if result.baseline_only_keys:
                print(f"      baseline-only keys: {result.baseline_only_keys}")
            if result.optimized_only_keys:
                print(f"      optimized-only keys: {result.optimized_only_keys}")
    
    return report


# =============================================================================
# Reporting
# =============================================================================

def print_summary(report: ValidationReport) -> None:
    """Print summary of validation results."""
    print()
    print("=" * 60)
    print("PAIR VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs:              {report.total_pairs}")
    print(f"Valid pairs:              {report.valid_pairs}")
    print(f"Signature mismatches:     {report.signature_mismatch_pairs}")
    print(f"Missing signatures:       {report.missing_signature_pairs}")
    print(f"Errors:                   {report.error_pairs}")
    
    if report.signature_mismatch_pairs > 0:
        print()
        print("PAIRS WITH SIGNATURE MISMATCHES:")
        for r in report.results:
            if r.mismatches:
                print(f"  {r.chapter}:{r.example_name}")
                for m in r.mismatches:
                    print(f"    {m.key}: baseline={m.baseline_value}, optimized={m.optimized_value}")


def generate_fix_suggestions(report: ValidationReport) -> List[str]:
    """Generate fix suggestions for mismatched pairs."""
    suggestions = []
    
    for r in report.results:
        if r.mismatches:
            suggestions.append(f"# {r.chapter}:{r.example_name}")
            suggestions.append(f"# Baseline: {r.baseline_path}")
            suggestions.append(f"# Optimized: {r.optimized_path}")
            suggestions.append("# Mismatches:")
            for m in r.mismatches:
                suggestions.append(f"#   {m.key}: baseline={m.baseline_value}, optimized={m.optimized_value}")
            suggestions.append("")
    
    return suggestions


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate baseline/optimized benchmark pairs have matching signatures"
    )
    parser.add_argument(
        "--chapter", "-c",
        help="Validate only a specific chapter (e.g., ch07, labs/moe_cuda)"
    )
    parser.add_argument(
        "--report", "-r",
        type=Path,
        help="Path to write JSON validation report"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=".",
        help="Root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    root_dir = args.root.resolve()
    
    # Run validation
    report = validate_all_pairs(root_dir, args.chapter)
    
    # Print summary
    print_summary(report)
    
    # Write report if requested
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport written to {args.report}")
    
    # Return non-zero if there are mismatches
    return 1 if report.signature_mismatch_pairs > 0 else 0


if __name__ == "__main__":
    sys.exit(main())



