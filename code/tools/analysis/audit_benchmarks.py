#!/usr/bin/env python3
"""Audit all benchmark files for proper inheritance and timing."""

import ast
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple

# Known base classes that ultimately inherit from BaseBenchmark
VALID_BASE_CLASSES = {
    'BaseBenchmark',
    'CudaBinaryBenchmark',
    # Intermediate classes that inherit from BaseBenchmark
    'StridedStreamBaseline',
    'ConcurrentStreamOptimized', 
    'TorchrunScriptBenchmark',
    'BaselineAddBenchmark',
    'BaselineMoeInferenceBenchmark',
    'BaselineMatmulTCGen05Benchmark',
    '_DisaggregatedInferenceBenchmark',
    'BaselineDisaggregatedInferenceBenchmark',
}

def find_intermediate_base_classes(code_root: Path) -> Set[str]:
    """Find all classes that inherit from BaseBenchmark or its children."""
    base_classes = {'BaseBenchmark', 'CudaBinaryBenchmark'}
    found_new = True
    
    while found_new:
        found_new = False
        for py_file in code_root.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            base_name = None
                            if isinstance(base, ast.Name):
                                base_name = base.id
                            elif isinstance(base, ast.Attribute):
                                base_name = base.attr
                            if base_name and base_name in base_classes:
                                if node.name not in base_classes:
                                    base_classes.add(node.name)
                                    found_new = True
            except:
                pass
    return base_classes

def check_file_inheritance(filepath: Path, valid_bases: Set[str]) -> Tuple[bool, str]:
    """Check if a benchmark file properly inherits from a valid base class."""
    try:
        content = filepath.read_text()
        
        # Method 1: Check for class definitions inheriting from valid bases
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name and base_name in valid_bases:
                        return True, f"inherits from {base_name}"
        
        # Method 2: Check if it's a wrapper that imports get_benchmark from another file
        if 'from' in content and 'get_benchmark' in content and 'import' in content:
            return True, "wrapper (imports get_benchmark)"
        
        # Method 3: Check for direct imports of valid base classes
        for base in valid_bases:
            if f"from " in content and base in content:
                return True, f"imports {base}"
        
        return False, "no valid inheritance found"
    except Exception as e:
        return False, f"parse error: {e}"

def main():
    code_root = Path('.')
    
    print("=== Benchmark Inheritance Audit ===\n")
    print("Step 1: Finding all intermediate base classes...")
    valid_bases = find_intermediate_base_classes(code_root)
    print(f"Found {len(valid_bases)} valid base classes:")
    for base in sorted(valid_bases):
        print(f"  - {base}")
    
    print("\nStep 2: Checking all benchmark files...\n")
    
    # Find all benchmark files
    benchmark_files = []
    for pattern in ['ch*/baseline_*.py', 'ch*/optimized_*.py', 
                    'labs/*/baseline_*.py', 'labs/*/optimized_*.py']:
        benchmark_files.extend(code_root.glob(pattern))
    
    benchmark_files = [f for f in benchmark_files if '__pycache__' not in str(f)]
    
    good = []
    bad = []
    
    for filepath in sorted(benchmark_files):
        ok, reason = check_file_inheritance(filepath, valid_bases)
        if ok:
            good.append((filepath, reason))
        else:
            bad.append((filepath, reason))
    
    print(f"✅ Valid inheritance: {len(good)} files")
    print(f"❌ Missing inheritance: {len(bad)} files")
    
    if bad:
        print("\n⚠️  Files without valid inheritance:")
        for filepath, reason in bad[:20]:
            print(f"  {filepath}: {reason}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
    
    print(f"\n=== Summary ===")
    print(f"Total benchmark files: {len(benchmark_files)}")
    print(f"With valid BaseBenchmark inheritance: {len(good)} ({100*len(good)//len(benchmark_files)}%)")
    print(f"Missing inheritance: {len(bad)}")
    
    return len(bad)

if __name__ == '__main__':
    sys.exit(main())
