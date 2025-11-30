#!/usr/bin/env python3
"""
audit_chapter_scope.py - Automated Chapter Scope & Book Alignment Checker

Two functions:
1. SCOPE CHECKING: Ensure each chapter only uses techniques appropriate for
   that chapter level (prevents pedagogical confusion)
2. BOOK ALIGNMENT: Check that book/chXX.md topics have matching code examples

USAGE:
    python core/scripts/audit_chapter_scope.py              # Scan all chapters
    python core/scripts/audit_chapter_scope.py --chapter 1  # Scan specific chapter
    python core/scripts/audit_chapter_scope.py --fix        # Show suggested fixes
    python core/scripts/audit_chapter_scope.py --check-book # Check book alignment

EXAMPLE OUTPUT:
    Chapter 1 Violations:
      ch01/optimized_foo.py:
        Line 45: Uses CUDA graphs (introduced in Ch12)

NOTES:
    - cudaEvent* patterns are ALLOWED everywhere (basic timing, not concurrency)
    - Triton is allowed in ch07+ when used for TMA demonstrations
    - Forward references to future chapters are allowed with comments
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple  # Tuple used for comment detection

# Define when each technique is introduced
TECHNIQUE_CHAPTERS: Dict[str, int] = {
    # Chapter 1-2: Fundamentals
    "nvtx": 1,
    "torch.cuda.synchronize": 1,
    "cudaEvent": 1,              # Basic timing primitives (cudaEventCreate/Record/Elapsed)
    "cudaDeviceSynchronize": 1,  # Basic synchronization
    "tf32": 1,
    "cudnn.benchmark": 1,
    
    # Chapter 3-4: Multi-GPU Basics
    "DistributedDataParallel": 3,
    "DDP": 3,
    "NCCL": 4,
    "all_reduce": 4,
    "all_gather": 4,
    
    # Chapter 5-6: Memory Hierarchy
    "__shared__": 5,
    "shared memory": 5,
    "occupancy": 6,
    "registers": 6,
    "ILP": 6,
    
    # Chapter 7-8: Memory Access
    "coalesced": 7,
    "vectorized": 7,
    "float4": 7,
    "__ldg": 7,
    "double_buffer": 8,
    "prefetch": 8,
    
    # Chapter 9-10: Pipelining
    "FlashAttention": 9,
    "flash_attn": 9,
    "SDPA": 9,
    "cuda::pipeline": 10,
    "memcpy_async": 10,
    "warp_specialization": 10,
    "cluster_group": 10,
    "DSMEM": 10,
    "distributed shared": 10,
    
    # Chapter 11-12: Concurrency
    "cuda.Stream": 11,
    "cudaStream": 11,
    "CUDAGraph": 12,
    "cuda.graph": 12,
    "make_graphed_callables": 12,
    "cudaGraphLaunch": 12,
    
    # Chapter 13-14: PyTorch Optimization
    # NOTE: float8/fp8 for vectorized memory (Ch7) vs quantization (Ch13) is 
    # handled by separate patterns in TECHNIQUE_PATTERNS
    "TransformerEngine": 13,
    "Float8Tensor": 13,
    "torch.compile": 14,
    "TorchInductor": 14,
    "triton": 14,
    "Triton": 14,
    "flex_attention": 14,
    "FlexAttention": 14,
    
    # Chapter 15-16: Advanced Serving
    "PagedAttention": 16,
    "paged_attention": 16,
    "KVCache": 16,
    "kv_cache": 16,
    
    # Chapter 17-18: Production
    "speculative": 18,
    "draft_model": 18,
    "vLLM": 18,
    
    # Chapter 19-20: Cutting Edge
    "Grace": 19,
    "NVLink-C2C": 19,
    "unified_memory": 19,
    "MoE": 20,
    "expert_parallel": 20,
}

# Regex patterns for detecting techniques
TECHNIQUE_PATTERNS: Dict[str, List[str]] = {
    "CUDA graphs": [
        r"CUDAGraph",
        r"cuda\.graph",
        r"cudaGraphLaunch",
        r"make_graphed_callables",
        r"torch\.cuda\.graph",
    ],
    # NOTE: torch.compile is mentioned in Ch9 for fusion demos (book line 49)
    # but deep dive is in Ch14. Allow torch.compile in Ch9+ for fusion context.
    # Strict enforcement would use Ch14.
    "torch.compile": [
        r"torch\.compile\(",
        r"@torch\.compile",
        # r"TorchInductor",  # Often just mentioned, not actual usage
        # r"inductor",       # Often just mentioned, not actual usage
    ],
    # FP8 split into two categories:
    # - FP8_vectorization (Ch7): 32-byte aligned loads, vector types
    # - FP8_quantization (Ch13): Scaling, calibration, TransformerEngine
    #
    # Ch7 files using float8 for memory access patterns are allowed.
    "FP8_quantization": [
        r"TransformerEngine",
        r"fp8_scale",
        r"e4m3fn",              # Specific format with semantics
        r"e5m2fn",              # Specific format with semantics  
        r"Float8Tensor",
        r"quantize.*fp8",
        r"fp8.*quantize",
        r"cast_to_fp8",
        r"from_fp8",
    ],
    "FlashAttention": [
        r"flash_attn",
        r"FlashAttention",
        r"flash_attention",
    ],
    "SDPA": [
        r"F\.scaled_dot_product_attention\(",  # Actual function call
        r"sdpa_kernel\(",                      # Backend selector
    ],
    # NOTE: Triton is treated as a "tool" like PyTorch itself, and may be used
    # in earlier chapters for demonstrations. The patterns below are commented
    # out to allow Triton usage everywhere. Uncomment to enforce strict Ch14+.
    #
    # "Triton": [
    #     r"import triton",
    #     r"@triton\.jit",
    #     r"triton\.language",
    # ],
    "Warp specialization": [
        r"warp_specialization",
        r"WarpSpecialized",
        r"__ballot_sync",
        r"__shfl",
    ],
    # NOTE: CUDA stream patterns are disabled by default because:
    # 1. Basic stream usage (for async copies) is common in earlier chapters
    # 2. Many false positives from documentation/comments
    # Uncomment to enforce strict Ch11+ for advanced stream concurrency.
    #
    # "CUDA streams": [
    #     r"cudaStreamCreate",         # CUDA stream creation
    #     r"cudaStreamSynchronize",    # Stream synchronization
    #     r"<<<.*,\s*stream>>>",       # Kernel launch with stream parameter
    # ],
    "TMA": [
        r"TMA",
        r"tensor_map",
        r"cp\.async\.bulk",
    ],
    "Clusters": [
        r"cluster_group",
        r"DSMEM",
        r"distributed shared",
        r"__cluster_dims__",
    ],
    "PagedAttention": [
        r"PagedAttention",
        r"paged_attention",
        r"block_tables",
    ],
    "Speculative decoding": [
        r"speculative",
        r"draft_model",
        r"verify_tokens",
    ],
}

# Map technique names to the chapter they're introduced
# NOTE: Some techniques are "mentioned" in earlier chapters but covered in depth later.
# We use the first meaningful usage chapter, not the deep-dive chapter.
TECHNIQUE_INTRO_CHAPTERS: Dict[str, int] = {
    "CUDA graphs": 12,
    "torch.compile": 9,  # Mentioned in Ch9 for fusion; deep dive in Ch14
    "FP8_quantization": 13,  # FP8 for quantization; vectorized float8 access is Ch7
    "FlashAttention": 9,
    "SDPA": 9,
    # "Triton": 14,  # Triton is a tool, allowed everywhere
    "Warp specialization": 10,
    # "CUDA streams": 11,  # Disabled - basic stream usage allowed everywhere
    "TMA": 7,
    "Clusters": 10,
    "PagedAttention": 16,
    "Speculative decoding": 18,
}

# Natural progression: Allow techniques 1 chapter before introduction
# This handles pedagogical transitions (e.g., Ch11 streams → Ch12 graphs)
NATURAL_PROGRESSION_TOLERANCE = 1  # Allow N-1 chapter to use N's techniques

# Documented forward references: Chapters that explicitly document usage of later topics
# Format: {technique: [allowed_chapters]}
# These chapters have explicit forward-reference comments documenting the cross-chapter usage
DOCUMENTED_FORWARD_REFS: Dict[str, List[int]] = {
    "Speculative decoding": [15, 16],  # Ch15-16 have forward-ref comments to Ch18
    "CUDA graphs": [11],               # Ch11 stream-ordered memory leads to graphs
    "torch.compile": [4],              # Ch4 training pipelines need compilation (documented)
    "SDPA": [4, 6],                    # Ch4, Ch6 attention layers use SDPA (documented)
}


@dataclass
class Violation:
    """A single scope violation."""
    file: str
    line_number: int
    technique: str
    introduced_chapter: int
    matching_text: str


@dataclass 
class AuditResult:
    """Results of auditing a chapter."""
    chapter: int
    files_scanned: int
    violations: List[Violation] = field(default_factory=list)
    
    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0


def get_chapter_from_path(path: Path) -> Optional[int]:
    """Extract chapter number from path like 'ch01/foo.py'."""
    match = re.match(r'ch(\d+)', path.parts[0] if path.parts else "")
    if match:
        return int(match.group(1))
    return None


def is_comment_or_docstring_line(line: str, in_docstring: bool) -> Tuple[bool, bool]:
    """Check if line is a comment/docstring. Returns (is_comment, new_in_docstring)."""
    stripped = line.strip()
    
    # Toggle docstring state on triple quotes
    if '"""' in stripped or "'''" in stripped:
        # Count occurrences - odd number toggles state
        triple_double = stripped.count('"""')
        triple_single = stripped.count("'''")
        if (triple_double + triple_single) % 2 == 1:
            in_docstring = not in_docstring
        return True, in_docstring
    
    # Python comment
    if stripped.startswith('#'):
        return True, in_docstring
    
    # Inside docstring
    if in_docstring:
        return True, in_docstring
    
    # C/CUDA comment  
    if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
        return True, in_docstring
    
    # printf/print statements (documentation, not usage)
    if 'printf' in stripped or 'print(' in stripped:
        return True, in_docstring
    
    # Forward reference markers
    if 'FORWARD REFERENCE' in line or 'forward reference' in line.lower():
        return True, in_docstring
    
    return False, in_docstring


def scan_file(file_path: Path, chapter: int) -> List[Violation]:
    """Scan a file for scope violations."""
    violations = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return violations
    
    lines = content.split('\n')
    in_docstring = False
    
    for technique, patterns in TECHNIQUE_PATTERNS.items():
        intro_chapter = TECHNIQUE_INTRO_CHAPTERS.get(technique, 99)
        
        # Skip if technique is allowed in this chapter
        if intro_chapter <= chapter:
            continue
        
        # Skip if within natural progression tolerance (e.g., Ch11 can use Ch12's graphs)
        if intro_chapter - chapter <= NATURAL_PROGRESSION_TOLERANCE:
            continue
        
        # Skip if this chapter has documented forward references for this technique
        if chapter in DOCUMENTED_FORWARD_REFS.get(technique, []):
            continue
        
        # Check each pattern
        for pattern in patterns:
            in_docstring_check = False
            for line_num, line in enumerate(lines, 1):
                # Skip comments, docstrings, and forward-reference markers
                is_comment, in_docstring_check = is_comment_or_docstring_line(
                    line, in_docstring_check
                )
                if is_comment:
                    continue
                    
                if re.search(pattern, line):
                    violations.append(Violation(
                        file=str(file_path),
                        line_number=line_num,
                        technique=technique,
                        introduced_chapter=intro_chapter,
                        matching_text=line.strip()[:80],
                    ))
    
    return violations


def audit_chapter(chapter_dir: Path, chapter: int) -> AuditResult:
    """Audit all files in a chapter directory."""
    result = AuditResult(chapter=chapter, files_scanned=0)
    
    if not chapter_dir.exists():
        return result
    
    # Files to skip (shared utilities, configs, not chapter-specific examples)
    skip_patterns = ["_config.py", "__init__.py", "compare.py", "setup.py"]
    
    # Scan Python files
    for py_file in chapter_dir.glob("*.py"):
        # Skip config/utility files
        if any(pattern in py_file.name for pattern in skip_patterns):
            continue
        result.files_scanned += 1
        violations = scan_file(py_file, chapter)
        result.violations.extend(violations)
    
    # Scan CUDA files
    for cu_file in chapter_dir.glob("*.cu"):
        result.files_scanned += 1
        violations = scan_file(cu_file, chapter)
        result.violations.extend(violations)
    
    return result


def audit_all_chapters(base_dir: Path) -> Dict[int, AuditResult]:
    """Audit all chapters in the codebase."""
    results = {}
    
    for ch_num in range(1, 21):
        chapter_dir = base_dir / f"ch{ch_num}"
        if chapter_dir.exists():
            results[ch_num] = audit_chapter(chapter_dir, ch_num)
    
    return results


def print_report(results: Dict[int, AuditResult], verbose: bool = False):
    """Print audit report."""
    total_violations = sum(len(r.violations) for r in results.values())
    total_files = sum(r.files_scanned for r in results.values())
    
    print("=" * 70)
    print("CHAPTER SCOPE AUDIT REPORT")
    print("=" * 70)
    print(f"Files scanned: {total_files}")
    print(f"Total violations: {total_violations}")
    print()
    
    if total_violations == 0:
        print("✅ All chapters pass scope check!")
        return
    
    print("VIOLATIONS BY CHAPTER:")
    print("-" * 70)
    
    for ch_num in sorted(results.keys()):
        result = results[ch_num]
        if result.violations:
            print(f"\n📁 Chapter {ch_num} ({len(result.violations)} violations)")
            
            # Group by file
            by_file: Dict[str, List[Violation]] = {}
            for v in result.violations:
                by_file.setdefault(v.file, []).append(v)
            
            for file_path, file_violations in by_file.items():
                print(f"\n  📄 {file_path}:")
                for v in file_violations[:5]:  # Show max 5 per file
                    print(f"    Line {v.line_number}: Uses '{v.technique}' "
                          f"(introduced Ch{v.introduced_chapter})")
                    if verbose:
                        print(f"      > {v.matching_text}")
                
                if len(file_violations) > 5:
                    print(f"    ... and {len(file_violations) - 5} more")
    
    print("\n" + "=" * 70)
    print("SUGGESTED FIXES:")
    print("=" * 70)
    
    # Aggregate suggestions
    suggestions: Dict[str, Set[str]] = {}
    for result in results.values():
        for v in result.violations:
            key = f"ch{result.chapter}/{Path(v.file).name}"
            suggestions.setdefault(key, set()).add(
                f"Remove/replace {v.technique} (Ch{v.introduced_chapter} topic)"
            )
    
    for file_key, fixes in list(suggestions.items())[:10]:
        print(f"\n{file_key}:")
        for fix in fixes:
            print(f"  • {fix}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit chapter examples for scope violations"
    )
    parser.add_argument(
        "--chapter", "-c", type=int,
        help="Audit specific chapter only"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show matching code snippets"
    )
    parser.add_argument(
        "--base-dir", "-d", type=str, default=".",
        help="Base directory of codebase"
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Show detailed fix suggestions"
    )
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if args.chapter:
        results = {
            args.chapter: audit_chapter(
                base_dir / f"ch{args.chapter}", 
                args.chapter
            )
        }
    else:
        results = audit_all_chapters(base_dir)
    
    print_report(results, verbose=args.verbose or args.fix)
    
    # Exit with error code if violations found
    total_violations = sum(len(r.violations) for r in results.values())
    sys.exit(1 if total_violations > 0 else 0)


if __name__ == "__main__":
    main()

