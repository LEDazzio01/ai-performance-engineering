#!/usr/bin/env python3
"""Audit script to find silent fallback patterns that should emit warnings.

This script identifies `except Exception: pass` patterns that silently hide
configuration failures. These should be replaced with explicit warnings.

Usage:
    python scripts/audit_silent_fallbacks.py           # Audit only
    python scripts/audit_silent_fallbacks.py --fix    # Apply fixes
    python scripts/audit_silent_fallbacks.py --dry-run # Show what would be fixed
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# Categories of silent fallbacks
CATEGORIES = {
    "precision": [
        r"torch\.set_float32_matmul_precision",
        r"\.half\(\)",
        r"\.bfloat16\(\)",
        r"dtype.*=.*torch\.float16",
    ],
    "sdpa_backend": [
        r"enable_flash_sdp",
        r"enable_math_sdp", 
        r"enable_mem_efficient_sdp",
        r"enable_cudnn_sdp",
    ],
    "compile": [
        r"torch\.compile",
        r"compile_fn\(",
        r"compile_model\(",
    ],
    "tma_config": [
        r"enable_tma",
        r"tma_support",
    ],
}

# Fix templates by category
FIX_TEMPLATES = {
    "precision": '''import warnings
try:
{try_body}
except Exception as e:
    warnings.warn(f"Precision configuration failed: {{e}}", RuntimeWarning)''',
    
    "sdpa_backend": '''import warnings
try:
{try_body}
except Exception as e:
    warnings.warn(f"SDPA backend configuration failed: {{e}}", RuntimeWarning)''',
    
    "compile": '''import warnings
try:
{try_body}
except Exception as e:
    warnings.warn(f"torch.compile failed, using eager mode: {{e}}", RuntimeWarning)''',
    
    "tma_config": '''import warnings
try:
{try_body}
except Exception as e:
    warnings.warn(f"TMA configuration failed: {{e}}", RuntimeWarning)''',
    
    "unknown": '''import warnings
try:
{try_body}
except Exception as e:
    warnings.warn(f"Configuration failed: {{e}}", RuntimeWarning)''',
}


def find_silent_fallbacks(filepath: Path) -> List[Tuple[int, str, str, int, int]]:
    """Find silent `except Exception: pass` patterns in a file.
    
    Returns list of (line_number, category, context, try_start, except_end) tuples.
    """
    try:
        content = filepath.read_text()
    except Exception:
        return []
    
    issues = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Look for `except Exception:` followed by `pass`
        if 'except Exception:' in line or 'except Exception as' in line:
            # Check next non-empty line for `pass`
            for j in range(i + 1, min(i + 3, len(lines))):
                next_line = lines[j].strip()
                if next_line == 'pass':
                    # Find the corresponding try block
                    try_start = None
                    indent = len(line) - len(line.lstrip())
                    for k in range(i - 1, -1, -1):
                        check_line = lines[k]
                        if check_line.strip().startswith('try:'):
                            check_indent = len(check_line) - len(check_line.lstrip())
                            if check_indent == indent:
                                try_start = k
                                break
                    
                    if try_start is None:
                        continue
                    
                    # Get context (the try block body)
                    context = '\n'.join(lines[try_start:i + 1])
                    
                    # Categorize the fallback
                    category = "unknown"
                    for cat, patterns in CATEGORIES.items():
                        for pattern in patterns:
                            if re.search(pattern, context, re.IGNORECASE):
                                category = cat
                                break
                        if category != "unknown":
                            break
                    
                    issues.append((i + 1, category, context.strip(), try_start, j))
                    break
                elif next_line and not next_line.startswith('#'):
                    break
    
    return issues


def fix_file(filepath: Path, issues: List[Tuple[int, str, str, int, int]], dry_run: bool = False) -> int:
    """Fix silent fallbacks in a file.
    
    Returns number of fixes applied.
    """
    if not issues:
        return 0
    
    content = filepath.read_text()
    lines = content.split('\n')
    
    # Sort issues by line number descending to avoid offset issues
    sorted_issues = sorted(issues, key=lambda x: x[3], reverse=True)
    
    fixes_applied = 0
    
    for line_no, category, context, try_start, except_end in sorted_issues:
        # Get the try block body (between try: and except)
        try_body_lines = []
        base_indent = len(lines[try_start]) - len(lines[try_start].lstrip())
        body_indent = base_indent + 4
        
        for k in range(try_start + 1, line_no - 1):  # line_no is 1-indexed, except line
            if lines[k].strip():  # Non-empty line
                try_body_lines.append(lines[k])
        
        if not try_body_lines:
            continue
        
        # Extract just the body content with proper indentation
        try_body = '\n'.join(try_body_lines)
        
        # Generate the fix
        template = FIX_TEMPLATES.get(category, FIX_TEMPLATES["unknown"])
        fix = template.format(try_body=try_body)
        
        # Apply proper indentation
        fix_lines = []
        for fix_line in fix.split('\n'):
            if fix_line.strip():
                fix_lines.append(' ' * base_indent + fix_line.lstrip())
            else:
                fix_lines.append('')
        
        if dry_run:
            print(f"\n  Would fix lines {try_start + 1}-{except_end + 1}:")
            print(f"  Before:")
            for k in range(try_start, except_end + 1):
                print(f"    {lines[k]}")
            print(f"  After:")
            for fl in fix_lines:
                print(f"    {fl}")
        else:
            # Replace the lines
            lines[try_start:except_end + 1] = fix_lines
        
        fixes_applied += 1
    
    if not dry_run and fixes_applied > 0:
        filepath.write_text('\n'.join(lines))
    
    return fixes_applied


def main() -> None:
    fix_mode = '--fix' in sys.argv
    dry_run = '--dry-run' in sys.argv
    
    repo_root = Path(__file__).parent.parent
    
    # Directories to search
    search_dirs = ['ch*/', 'labs/', 'common/', 'tools/']
    
    total_issues = 0
    files_with_issues = 0
    fixes_applied = 0
    category_counts: dict[str, int] = {}
    
    for pattern in search_dirs:
        for filepath in repo_root.glob(f'{pattern}**/*.py'):
            if filepath.is_file():
                issues = find_silent_fallbacks(filepath)
                if issues:
                    files_with_issues += 1
                    print(f"\n{filepath.relative_to(repo_root)}:")
                    
                    for line_no, category, context, try_start, except_end in issues:
                        total_issues += 1
                        category_counts[category] = category_counts.get(category, 0) + 1
                        print(f"  Line {line_no} [{category}]:")
                        # Show just first 2 lines of context
                        context_lines = context.split('\n')[-2:]
                        for ctx_line in context_lines:
                            print(f"    {ctx_line}")
                    
                    if fix_mode or dry_run:
                        fixed = fix_file(filepath, issues, dry_run=dry_run)
                        fixes_applied += fixed
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {total_issues} silent fallbacks in {files_with_issues} files")
    print(f"\nBy category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    if fix_mode:
        print(f"\n✅ Applied {fixes_applied} fixes")
    elif dry_run:
        print(f"\n📋 Would apply {fixes_applied} fixes (dry-run mode)")
    else:
        print(f"\nRecommendation: REMOVE try/except blocks - use fail-fast:")
        print("""
    # DON'T do this (AI slop):
    try:
        do_thing()
    except Exception:
        pass
    
    # DO this instead (emit warning):
    import warnings
    try:
        do_thing()
    except Exception as e:
        warnings.warn(f"do_thing failed: {e}", RuntimeWarning)
    
    # OR better - just let it fail:
    do_thing()  # Let it fail if something is wrong
        """)
        print("\nRun with --fix to apply automatic fixes, or --dry-run to preview.")
        print("  python scripts/audit_silent_fallbacks.py --dry-run")
        print("  python scripts/audit_silent_fallbacks.py --fix")


if __name__ == '__main__':
    main()

