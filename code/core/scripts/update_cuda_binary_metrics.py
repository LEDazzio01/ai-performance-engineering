#!/usr/bin/env python3
"""Update CudaBinaryBenchmark wrappers with domain-specific get_custom_metrics().

These files already inherit from CudaBinaryBenchmark which now has a base
get_custom_metrics() implementation. This script adds domain-specific overrides.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Domain-specific metrics for CudaBinaryBenchmark wrappers
DOMAIN_METRICS = {
    # Ch7: Memory access patterns
    "ch07": {
        "domain": "memory_access",
        "baseline_metrics": {
            "is_coalesced": "0.0",
            "expected_efficiency_pct": "3.125",  # 1/32 for stride access
        },
        "optimized_metrics": {
            "is_coalesced": "1.0",
            "expected_efficiency_pct": "100.0",
        },
    },
    # Ch8: Optimization techniques
    "ch08": {
        "domain": "optimization",
        "baseline_metrics": {
            "has_optimization": "0.0",
        },
        "optimized_metrics": {
            "has_optimization": "1.0",
        },
    },
    # Ch9: Compute-bound / CUTLASS
    "ch09": {
        "domain": "compute",
        "baseline_metrics": {
            "uses_tensor_cores": "0.0",
            "uses_cutlass": "0.0",
        },
        "optimized_metrics": {
            "uses_tensor_cores": "1.0",
            "uses_cutlass": "1.0",
        },
    },
    # Ch10: Pipelines and clusters  
    "ch10": {
        "domain": "pipeline",
        "baseline_metrics": {
            "uses_clusters": "0.0",
            "uses_pipeline": "0.0",
        },
        "optimized_metrics": {
            "uses_clusters": "1.0",
            "uses_pipeline": "1.0",
        },
    },
    # Ch11: Streams
    "ch11": {
        "domain": "stream",
        "baseline_metrics": {
            "has_overlap": "0.0",
        },
        "optimized_metrics": {
            "has_overlap": "1.0",
        },
    },
    # Ch12: CUDA Graphs
    "ch12": {
        "domain": "graph",
        "baseline_metrics": {
            "uses_graph": "0.0",
        },
        "optimized_metrics": {
            "uses_graph": "1.0",
        },
    },
}


def get_chapter(filepath: Path) -> str:
    """Extract chapter from filepath."""
    for part in filepath.parts:
        if part.startswith("ch") and len(part) <= 4:
            return part
    return ""


def is_baseline(filepath: Path) -> bool:
    """Check if this is a baseline file."""
    return "baseline" in filepath.name


def has_custom_metrics_override(content: str) -> bool:
    """Check if file already overrides get_custom_metrics."""
    # Look for the method definition at the class level (indented)
    # Exclude the base class definition
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '    def get_custom_metrics' in line and i > 0:
            # Check if previous lines have a class definition that's not CudaBinaryBenchmark
            for j in range(i - 1, max(0, i - 20), -1):
                if 'class ' in lines[j] and 'CudaBinaryBenchmark' not in lines[j]:
                    return True
                if 'class ' in lines[j] and 'CudaBinaryBenchmark' in lines[j]:
                    # Check if the get_custom_metrics is inside this class
                    return '    def get_custom_metrics' in content
    return False


def add_metrics_override(filepath: Path, dry_run: bool = True) -> bool:
    """Add domain-specific get_custom_metrics override to a file."""
    content = filepath.read_text()
    
    # Skip if already has override
    if 'def get_custom_metrics' in content:
        return False
    
    # Get chapter and determine metrics
    chapter = get_chapter(filepath)
    if chapter not in DOMAIN_METRICS:
        return False
    
    domain_info = DOMAIN_METRICS[chapter]
    domain = domain_info["domain"]
    metrics = domain_info["baseline_metrics"] if is_baseline(filepath) else domain_info["optimized_metrics"]
    
    # Build the metrics override method
    metrics_lines = []
    for key, value in metrics.items():
        metrics_lines.append(f'            "{domain}.{key}": {value},')
    
    method_code = f'''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific {domain} metrics."""
        base_metrics = super().get_custom_metrics() or {{}}
        base_metrics.update({{
{chr(10).join(metrics_lines)}
    }})
        return base_metrics
'''
    
    # Find insertion point - after __init__ method
    # Look for the end of __init__ (next method or class end)
    init_match = re.search(r'def __init__\(self\)[^:]*:.*?(?=\n    def |\nclass |\ndef [a-z]|\Z)', 
                          content, re.DOTALL)
    if not init_match:
        return False
    
    insertion_point = init_match.end()
    
    # Add Optional import if needed
    if 'from typing import' in content and 'Optional' not in content:
        content = re.sub(r'(from typing import)', r'\1 Optional,', content)
    elif 'from typing import' not in content:
        # Add import after __future__ imports
        future_match = re.search(r'from __future__ import.*\n', content)
        if future_match:
            content = content[:future_match.end()] + 'from typing import Optional\n' + content[future_match.end():]
            insertion_point += len('from typing import Optional\n')
    
    new_content = content[:insertion_point] + method_code + content[insertion_point:]
    
    if dry_run:
        print(f"  WOULD ADD {domain} metrics to {filepath.name}")
        return True
    
    filepath.write_text(new_content)
    print(f"  ADDED {domain} metrics to {filepath.name}")
    return True


def find_cuda_binary_files() -> list:
    """Find all CudaBinaryBenchmark wrapper files."""
    files = []
    for pattern in ["ch*/*.py", "labs/*/*.py"]:
        for f in REPO_ROOT.glob(pattern):
            if "__pycache__" in str(f):
                continue
                content = f.read_text()
                if "CudaBinaryBenchmark" in content and "class " in content:
                    files.append(f)
    return sorted(files)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    files = find_cuda_binary_files()
    print(f"Found {len(files)} CudaBinaryBenchmark wrapper files")
    
    modified = 0
    skipped = 0
    
    for f in files:
        chapter = get_chapter(f)
        if chapter not in DOMAIN_METRICS:
            skipped += 1
            continue
        
        content = f.read_text()
        if 'def get_custom_metrics' in content:
            skipped += 1
            continue
        
        if add_metrics_override(f, dry_run=args.dry_run):
            modified += 1
    
    print(f"\nSummary:")
    print(f"  Modified (or would modify): {modified}")
    print(f"  Skipped (already has or no domain mapping): {skipped}")


if __name__ == "__main__":
    main()

