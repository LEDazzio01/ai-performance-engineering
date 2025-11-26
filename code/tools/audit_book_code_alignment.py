#!/usr/bin/env python3
"""
audit_book_code_alignment.py - Book ↔ Code Alignment Checker

Parses book/chXX.md files for code snippets and verifies that corresponding
implementations exist in chXX/ directories.

USAGE:
    python tools/audit_book_code_alignment.py              # Check all chapters
    python tools/audit_book_code_alignment.py --chapter 9  # Check specific chapter
    python tools/audit_book_code_alignment.py --verbose    # Show all snippets

WHAT IT CHECKS:
    1. Code snippets in book have matching files in code directory
    2. Key functions/classes mentioned in book exist in code
    3. Technique coverage (book topics vs code examples)

EXAMPLE OUTPUT:
    Chapter 9: 12 code snippets found
      ✅ fusedL2Norm kernel → ch9/baseline_fused_l2norm.cu
      ✅ CUTLASS GEMM → ch9/optimized_cutlass_gemm.py
      ⚠️ No match: flashAttentionKernel
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Patterns to extract from book code snippets
CODE_PATTERNS = {
    # CUDA kernels
    "cuda_kernel": re.compile(r"__global__\s+void\s+(\w+)"),
    # Python functions
    "python_func": re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE),
    # Python classes
    "python_class": re.compile(r"^class\s+(\w+)", re.MULTILINE),
    # Triton kernels
    "triton_kernel": re.compile(r"@triton\.jit\s*\n.*?def\s+(\w+)", re.DOTALL),
    # CUDA device functions
    "cuda_device": re.compile(r"__device__\s+\w+\s+(\w+)\s*\("),
    # PyTorch modules
    "torch_module": re.compile(r"class\s+(\w+)\s*\(\s*nn\.Module\s*\)"),
}

# Key technique keywords to look for in book
TECHNIQUE_KEYWORDS = {
    "ch7": ["coalesced", "vectorized", "float4", "TMA", "prefetch", "cache"],
    "ch8": ["double_buffer", "pipelining", "async", "overlap"],
    "ch9": ["fusion", "fused", "tiling", "arithmetic_intensity", "SDPA", "flash"],
    "ch10": ["warp_specialization", "cluster", "DSMEM", "producer", "consumer"],
    "ch11": ["stream", "concurrent", "overlap", "async"],
    "ch12": ["graph", "CUDAGraph", "capture", "replay"],
    "ch13": ["fp8", "FP8", "quantization", "mixed_precision"],
    "ch14": ["compile", "triton", "inductor", "flex_attention"],
    "ch18": ["speculative", "vllm", "draft", "paged_attention"],
}


@dataclass
class CodeSnippet:
    """A code snippet extracted from the book."""
    language: str  # cpp, python, cuda, etc.
    content: str
    line_number: int
    identifiers: List[str] = field(default_factory=list)  # Functions, classes, kernels


@dataclass
class AlignmentResult:
    """Result of checking book-code alignment for a chapter."""
    chapter: int
    snippets_found: int
    identifiers_found: List[str]
    matched_files: Dict[str, str]  # identifier -> file path
    unmatched: List[str]
    technique_coverage: Dict[str, bool]  # keyword -> found in code


def extract_code_snippets(book_path: Path) -> List[CodeSnippet]:
    """Extract code snippets from a markdown book chapter."""
    snippets = []
    
    try:
        content = book_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not read {book_path}: {e}", file=sys.stderr)
        return snippets
    
    # Find all code blocks (``` ... ```)
    # Pattern matches ```language\ncode\n```
    code_block_pattern = re.compile(
        r"```(\w*)\n(.*?)```",
        re.DOTALL
    )
    
    lines = content.split('\n')
    line_offsets = {}
    offset = 0
    for i, line in enumerate(lines, 1):
        line_offsets[offset] = i
        offset += len(line) + 1
    
    for match in code_block_pattern.finditer(content):
        language = match.group(1).lower() or "unknown"
        code_content = match.group(2)
        
        # Find line number
        start_pos = match.start()
        line_num = 1
        for offset, ln in sorted(line_offsets.items()):
            if offset <= start_pos:
                line_num = ln
            else:
                break
        
        # Extract identifiers from the code
        identifiers = []
        for pattern_name, pattern in CODE_PATTERNS.items():
            for id_match in pattern.finditer(code_content):
                identifier = id_match.group(1)
                if identifier and len(identifier) > 2:  # Skip short names
                    identifiers.append(identifier)
        
        if code_content.strip():  # Only add non-empty snippets
            snippets.append(CodeSnippet(
                language=language,
                content=code_content[:200],  # Truncate for display
                line_number=line_num,
                identifiers=list(set(identifiers)),  # Dedupe
            ))
    
    return snippets


def search_identifier_in_code(identifier: str, code_dir: Path) -> Optional[str]:
    """Search for an identifier in the code directory."""
    # Normalize identifier for file matching
    identifier_lower = identifier.lower()
    
    # Check Python files
    for py_file in code_dir.glob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            # Look for the identifier as a function, class, or in filename
            if identifier in content or identifier_lower in py_file.stem.lower():
                return str(py_file.relative_to(code_dir.parent))
        except Exception:
            continue
    
    # Check CUDA files
    for cu_file in code_dir.glob("*.cu"):
        try:
            content = cu_file.read_text(encoding='utf-8')
            if identifier in content or identifier_lower in cu_file.stem.lower():
                return str(cu_file.relative_to(code_dir.parent))
        except Exception:
            continue
    
    return None


def check_technique_coverage(chapter: int, code_dir: Path) -> Dict[str, bool]:
    """Check if technique keywords from book are covered in code."""
    coverage = {}
    ch_key = f"ch{chapter}"
    
    if ch_key not in TECHNIQUE_KEYWORDS:
        return coverage
    
    # Read all code files
    all_code = ""
    for py_file in code_dir.glob("*.py"):
        try:
            all_code += py_file.read_text(encoding='utf-8').lower() + "\n"
        except Exception:
            continue
    for cu_file in code_dir.glob("*.cu"):
        try:
            all_code += cu_file.read_text(encoding='utf-8').lower() + "\n"
        except Exception:
            continue
    
    # Check each keyword
    for keyword in TECHNIQUE_KEYWORDS[ch_key]:
        coverage[keyword] = keyword.lower() in all_code
    
    return coverage


def audit_chapter(chapter: int, base_dir: Path) -> AlignmentResult:
    """Audit book-code alignment for a single chapter."""
    ch_pad = f"{chapter:02d}"
    book_path = base_dir / "book" / f"ch{ch_pad}.md"
    code_dir = base_dir / f"ch{chapter}"
    
    result = AlignmentResult(
        chapter=chapter,
        snippets_found=0,
        identifiers_found=[],
        matched_files={},
        unmatched=[],
        technique_coverage={},
    )
    
    if not book_path.exists():
        return result
    
    # Extract code snippets from book
    snippets = extract_code_snippets(book_path)
    result.snippets_found = len(snippets)
    
    # Collect all identifiers
    all_identifiers = set()
    for snippet in snippets:
        all_identifiers.update(snippet.identifiers)
    
    result.identifiers_found = sorted(all_identifiers)
    
    # Check if identifiers exist in code
    if code_dir.exists():
        for identifier in all_identifiers:
            match = search_identifier_in_code(identifier, code_dir)
            if match:
                result.matched_files[identifier] = match
            else:
                result.unmatched.append(identifier)
        
        # Check technique coverage
        result.technique_coverage = check_technique_coverage(chapter, code_dir)
    
    return result


def print_report(results: Dict[int, AlignmentResult], verbose: bool = False):
    """Print the alignment report."""
    print("=" * 70)
    print("BOOK ↔ CODE ALIGNMENT REPORT")
    print("=" * 70)
    
    total_snippets = sum(r.snippets_found for r in results.values())
    total_identifiers = sum(len(r.identifiers_found) for r in results.values())
    total_matched = sum(len(r.matched_files) for r in results.values())
    total_unmatched = sum(len(r.unmatched) for r in results.values())
    
    print(f"Total code snippets in book: {total_snippets}")
    print(f"Total identifiers extracted: {total_identifiers}")
    print(f"Matched in code: {total_matched}")
    print(f"Unmatched: {total_unmatched}")
    print()
    
    for ch_num in sorted(results.keys()):
        result = results[ch_num]
        if result.snippets_found == 0:
            continue
        
        match_rate = len(result.matched_files) / max(len(result.identifiers_found), 1) * 100
        
        print(f"\n📖 Chapter {ch_num}: {result.snippets_found} snippets, "
              f"{len(result.identifiers_found)} identifiers ({match_rate:.0f}% matched)")
        print("-" * 50)
        
        # Show matched
        if result.matched_files and verbose:
            print("  ✅ Matched:")
            for ident, file_path in sorted(result.matched_files.items())[:10]:
                print(f"     {ident} → {file_path}")
            if len(result.matched_files) > 10:
                print(f"     ... and {len(result.matched_files) - 10} more")
        
        # Show unmatched
        if result.unmatched:
            # Filter out common/generic names
            meaningful_unmatched = [
                u for u in result.unmatched 
                if u not in ("main", "init", "forward", "setup", "run", "test")
                and not u.startswith("__")
                and len(u) > 3
            ]
            if meaningful_unmatched:
                print(f"  ⚠️ Unmatched ({len(meaningful_unmatched)}):")
                for ident in sorted(meaningful_unmatched)[:8]:
                    print(f"     {ident}")
                if len(meaningful_unmatched) > 8:
                    print(f"     ... and {len(meaningful_unmatched) - 8} more")
        
        # Show technique coverage
        if result.technique_coverage:
            covered = [k for k, v in result.technique_coverage.items() if v]
            missing = [k for k, v in result.technique_coverage.items() if not v]
            
            if covered:
                print(f"  📝 Techniques covered: {', '.join(covered)}")
            if missing:
                print(f"  ❌ Techniques missing: {', '.join(missing)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    low_coverage = [
        ch for ch, r in results.items() 
        if r.identifiers_found and len(r.matched_files) / len(r.identifiers_found) < 0.5
    ]
    
    if low_coverage:
        print(f"⚠️ Chapters with <50% identifier match: {low_coverage}")
    else:
        print("✅ All chapters have good book-code alignment!")


def main():
    parser = argparse.ArgumentParser(
        description="Check alignment between book chapters and code examples"
    )
    parser.add_argument(
        "--chapter", "-c", type=int,
        help="Check specific chapter only"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed matches"
    )
    parser.add_argument(
        "--base-dir", "-d", type=str, default=".",
        help="Base directory of codebase"
    )
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if args.chapter:
        results = {args.chapter: audit_chapter(args.chapter, base_dir)}
    else:
        results = {}
        for ch_num in range(1, 21):
            results[ch_num] = audit_chapter(ch_num, base_dir)
    
    print_report(results, verbose=args.verbose)


if __name__ == "__main__":
    main()

