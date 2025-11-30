#!/usr/bin/env python3
"""Find benchmark pairs that lack input signatures for verification.

This script identifies benchmarks that would fail input verification
because they don't expose workload attributes.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.discovery import discover_benchmark_pairs


def load_benchmark_from_file(file_path: Path) -> Optional[Any]:
    """Load a benchmark from a file without running it."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"benchmark_{file_path.stem}",
            file_path
        )
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        if hasattr(module, 'get_benchmark'):
            return module.get_benchmark()
        return None
    except Exception as e:
        return None


def check_input_signature(benchmark) -> Tuple[bool, Dict[str, Any]]:
    """Check if a benchmark has a non-empty input signature."""
    if benchmark is None:
        return False, {}
    
    try:
        sig_fn = getattr(benchmark, 'get_input_signature', None)
        if sig_fn and callable(sig_fn):
            signature = sig_fn()
            if signature:
                return True, signature
    except Exception:
        pass
    
    return False, {}


def main():
    """Find all benchmark pairs lacking input signatures."""
    pairs = discover_benchmark_pairs(REPO_ROOT, chapter="all")
    
    print(f"Found {len(pairs)} benchmark pairs total\n")
    
    missing_signatures: List[Tuple[Path, List[Path], str]] = []
    has_signatures: List[Tuple[Path, List[Path], str, Dict]] = []
    errors: List[Tuple[Path, str]] = []
    
    for baseline_path, optimized_paths, example_name in pairs:
        try:
            baseline = load_benchmark_from_file(baseline_path)
            has_sig, sig = check_input_signature(baseline)
            
            if has_sig:
                has_signatures.append((baseline_path, optimized_paths, example_name, sig))
            else:
                missing_signatures.append((baseline_path, optimized_paths, example_name))
        except Exception as e:
            errors.append((baseline_path, str(e)))
    
    # Print summary
    print("=" * 80)
    print(f"SUMMARY: {len(missing_signatures)} pairs LACK input signatures")
    print(f"         {len(has_signatures)} pairs HAVE input signatures")
    print(f"         {len(errors)} pairs had errors")
    print("=" * 80)
    
    if missing_signatures:
        print("\n" + "=" * 80)
        print("PAIRS MISSING INPUT SIGNATURES (need get_input_signature()):")
        print("=" * 80)
        
        # Group by chapter
        by_chapter: Dict[str, List[Tuple[Path, List[Path], str]]] = {}
        for baseline_path, optimized_paths, example_name in missing_signatures:
            chapter = baseline_path.parent.name
            if chapter not in by_chapter:
                by_chapter[chapter] = []
            by_chapter[chapter].append((baseline_path, optimized_paths, example_name))
        
        for chapter in sorted(by_chapter.keys()):
            items = by_chapter[chapter]
            print(f"\n{chapter}/ ({len(items)} pairs):")
            for baseline_path, optimized_paths, example_name in items:
                opt_names = [p.name for p in optimized_paths]
                print(f"  - {baseline_path.name} -> {', '.join(opt_names)}")
    
    if has_signatures:
        print("\n" + "=" * 80)
        print("PAIRS WITH INPUT SIGNATURES (good):")
        print("=" * 80)
        
        # Group by chapter
        by_chapter: Dict[str, List] = {}
        for baseline_path, optimized_paths, example_name, sig in has_signatures:
            chapter = baseline_path.parent.name
            if chapter not in by_chapter:
                by_chapter[chapter] = []
            by_chapter[chapter].append((baseline_path, optimized_paths, example_name, sig))
        
        for chapter in sorted(by_chapter.keys()):
            items = by_chapter[chapter]
            print(f"\n{chapter}/ ({len(items)} pairs):")
            for baseline_path, optimized_paths, example_name, sig in items:
                sig_keys = list(sig.keys())[:5]  # Show first 5 keys
                sig_preview = ', '.join(f"{k}={sig[k]}" for k in sig_keys)
                if len(sig.keys()) > 5:
                    sig_preview += f", ... (+{len(sig.keys()) - 5} more)"
                print(f"  - {example_name}: {sig_preview}")
    
    if errors:
        print("\n" + "=" * 80)
        print("ERRORS:")
        print("=" * 80)
        for path, error in errors[:10]:
            print(f"  - {path}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Generate actionable output
    print("\n" + "=" * 80)
    print("FILES THAT NEED get_input_signature():")
    print("=" * 80)
    for baseline_path, _, _ in missing_signatures:
        print(baseline_path)
    
    return len(missing_signatures)


if __name__ == "__main__":
    missing_count = main()
    sys.exit(0 if missing_count == 0 else 1)








