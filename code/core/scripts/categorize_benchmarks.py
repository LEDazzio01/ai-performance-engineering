#!/usr/bin/env python3
"""Categorize all benchmarks by type for migration planning.

This script categorizes benchmarks into:
- output_tensor: Has self.output, self.C, etc. → auto-generate get_verify_output()
- training_loop: Has optimizer/loss → add exemption + loss checksum
- throughput_only: Bandwidth/latency tests → add exemption with justification
- cuda_binary: CudaBinaryBenchmark → manual VERIFY_CHECKSUM handling
- already_compliant: Has all verification methods

Usage:
    python -m core.scripts.categorize_benchmarks
    python -m core.scripts.categorize_benchmarks --output artifacts/benchmark_categories.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Category Definitions
# =============================================================================

CATEGORIES = {
    "output_tensor": "Has output tensor (self.output, self.C, etc.)",
    "training_loop": "Training loop with optimizer/loss",
    "throughput_only": "Throughput/bandwidth-only benchmark",
    "cuda_binary": "CudaBinaryBenchmark subclass",
    "already_compliant": "Has all verification methods",
    "parse_error": "Could not parse file",
    "no_benchmark_class": "No benchmark class found",
}

# Attributes that indicate output tensors
OUTPUT_ATTRIBUTES = {
    "output", "out", "C", "result", "results", "y",
    "logits", "hidden_states", "attention_output",
}

# Attributes that indicate training loop
TRAINING_ATTRIBUTES = {
    "optimizer", "loss", "loss_fn", "criterion",
    "scheduler", "lr_scheduler", "grad", "backward",
}

# Keywords in filename/class that indicate throughput-only
THROUGHPUT_KEYWORDS = {
    "bandwidth", "throughput", "latency", "memory_test",
    "memcpy", "copy", "transfer", "nvlink", "pcie",
    "collective", "allreduce", "allgather", "nccl",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkInfo:
    """Information about a single benchmark."""
    file_path: str
    benchmark_class: Optional[str] = None
    parent_class: Optional[str] = None
    category: str = "unknown"
    
    # Compliance status
    has_get_input_signature: bool = False
    has_validate_result: bool = False
    has_get_workload_metadata: bool = False
    has_get_verify_output: bool = False
    
    # Detected attributes
    output_attrs: Set[str] = field(default_factory=set)
    training_attrs: Set[str] = field(default_factory=set)
    
    # Errors
    error: Optional[str] = None


@dataclass
class CategorizationReport:
    """Report of benchmark categorization."""
    total_files: int = 0
    by_category: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    by_chapter: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    benchmarks: List[BenchmarkInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "summary": {cat: len(files) for cat, files in self.by_category.items()},
            "by_category": {cat: files for cat, files in self.by_category.items()},
            "by_chapter": {ch: dict(cats) for ch, cats in self.by_chapter.items()},
            "benchmarks": [
                {
                    "file_path": b.file_path,
                    "benchmark_class": b.benchmark_class,
                    "parent_class": b.parent_class,
                    "category": b.category,
                    "has_get_input_signature": b.has_get_input_signature,
                    "has_validate_result": b.has_validate_result,
                    "has_get_workload_metadata": b.has_get_workload_metadata,
                    "has_get_verify_output": b.has_get_verify_output,
                    "output_attrs": list(b.output_attrs),
                    "training_attrs": list(b.training_attrs),
                    "error": b.error,
                }
                for b in self.benchmarks
            ],
        }


# =============================================================================
# AST Analysis
# =============================================================================

class BenchmarkAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze benchmark files."""
    
    def __init__(self):
        self.benchmark_class: Optional[str] = None
        self.parent_class: Optional[str] = None
        self.methods: Set[str] = set()
        self.output_attrs: Set[str] = set()
        self.training_attrs: Set[str] = set()
        self.all_attrs: Set[str] = set()
        self._in_benchmark_class = False
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to find benchmark classes."""
        has_benchmark_fn = any(
            isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
            for item in node.body
        )
        
        if has_benchmark_fn:
            self.benchmark_class = node.name
            
            if node.bases:
                base = node.bases[0]
                if isinstance(base, ast.Name):
                    self.parent_class = base.id
                elif isinstance(base, ast.Attribute):
                    self.parent_class = base.attr
            
            self._in_benchmark_class = True
            self.generic_visit(node)
            self._in_benchmark_class = False
        else:
            self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to detect methods."""
        if self._in_benchmark_class:
            self.methods.add(node.name)
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to detect attributes."""
        if not self._in_benchmark_class:
            self.generic_visit(node)
            return
        
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    attr_name = target.attr
                    self.all_attrs.add(attr_name)
                    
                    if attr_name in OUTPUT_ATTRIBUTES:
                        self.output_attrs.add(attr_name)
                    
                    if attr_name in TRAINING_ATTRIBUTES:
                        self.training_attrs.add(attr_name)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to detect training patterns."""
        if self._in_benchmark_class:
            # Check for loss.backward(), optimizer.step(), etc.
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("backward", "step", "zero_grad"):
                    self.training_attrs.add(node.func.attr)
        
        self.generic_visit(node)


def categorize_benchmark(file_path: Path) -> BenchmarkInfo:
    """Categorize a single benchmark file."""
    info = BenchmarkInfo(file_path=str(file_path))
    
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        info.category = "parse_error"
        info.error = f"Syntax error: {e}"
        return info
    except Exception as e:
        info.category = "parse_error"
        info.error = f"Parse error: {e}"
        return info
    
    analyzer = BenchmarkAnalyzer()
    analyzer.visit(tree)
    
    if not analyzer.benchmark_class:
        info.category = "no_benchmark_class"
        return info
    
    info.benchmark_class = analyzer.benchmark_class
    info.parent_class = analyzer.parent_class
    info.has_get_input_signature = "get_input_signature" in analyzer.methods
    info.has_validate_result = "validate_result" in analyzer.methods
    info.has_get_workload_metadata = "get_workload_metadata" in analyzer.methods
    info.has_get_verify_output = "get_verify_output" in analyzer.methods
    info.output_attrs = analyzer.output_attrs
    info.training_attrs = analyzer.training_attrs
    
    # Determine category
    
    # 1. Check if already compliant
    if (info.has_get_input_signature and 
        info.has_validate_result and 
        info.has_get_workload_metadata):
        info.category = "already_compliant"
        return info
    
    # 2. Check if CUDA binary
    if analyzer.parent_class == "CudaBinaryBenchmark":
        info.category = "cuda_binary"
        return info
    
    # 3. Check if training loop
    if analyzer.training_attrs:
        info.category = "training_loop"
        return info
    
    # 4. Check filename/class for throughput keywords
    name_lower = (file_path.name + (analyzer.benchmark_class or "")).lower()
    if any(kw in name_lower for kw in THROUGHPUT_KEYWORDS):
        info.category = "throughput_only"
        return info
    
    # 5. Check if has output tensor
    if analyzer.output_attrs:
        info.category = "output_tensor"
        return info
    
    # 6. Default to throughput_only if no output detected
    info.category = "throughput_only"
    return info


def find_benchmark_files(root_dir: Path) -> List[Path]:
    """Find all benchmark files."""
    files: List[Path] = []
    patterns = ["baseline_*.py", "optimized_*.py"]
    
    # Find chapter directories
    chapter_dirs = [
        d for d in root_dir.iterdir()
        if d.is_dir() and d.name.startswith("ch")
    ]
    
    # Add labs directories
    labs_dir = root_dir / "labs"
    if labs_dir.exists():
        chapter_dirs.extend(d for d in labs_dir.iterdir() if d.is_dir())
    
    for chapter_dir in chapter_dirs:
        for pattern in patterns:
            files.extend(chapter_dir.glob(pattern))
            files.extend(chapter_dir.glob(f"**/{pattern}"))
    
    return sorted(set(files))


def categorize_all(root_dir: Path) -> CategorizationReport:
    """Categorize all benchmarks in the repository."""
    report = CategorizationReport()
    
    files = find_benchmark_files(root_dir)
    report.total_files = len(files)
    
    print(f"Found {len(files)} benchmark files")
    
    for file_path in files:
        info = categorize_benchmark(file_path)
        report.benchmarks.append(info)
        report.by_category[info.category].append(str(file_path))
        
        # Get chapter name
        chapter = file_path.parent.name
        if "labs/" in str(file_path):
            chapter = f"labs/{chapter}"
        report.by_chapter[chapter][info.category] += 1
    
    return report


def print_report(report: CategorizationReport) -> None:
    """Print a human-readable report."""
    print("\n" + "=" * 70)
    print("BENCHMARK CATEGORIZATION REPORT")
    print("=" * 70)
    
    print("\n## Summary by Category\n")
    print(f"{'Category':<25} {'Count':>8} {'Description'}")
    print("-" * 70)
    for cat, description in CATEGORIES.items():
        count = len(report.by_category[cat])
        print(f"{cat:<25} {count:>8}   {description}")
    print(f"{'TOTAL':<25} {report.total_files:>8}")
    
    print("\n## Summary by Chapter\n")
    print(f"{'Chapter':<20} {'output':>8} {'training':>8} {'thruput':>8} {'cuda':>8} {'compliant':>8} {'error':>8}")
    print("-" * 80)
    for chapter in sorted(report.by_chapter.keys()):
        cats = report.by_chapter[chapter]
        print(f"{chapter:<20} {cats.get('output_tensor', 0):>8} {cats.get('training_loop', 0):>8} "
              f"{cats.get('throughput_only', 0):>8} {cats.get('cuda_binary', 0):>8} "
              f"{cats.get('already_compliant', 0):>8} {cats.get('parse_error', 0):>8}")
    
    print("\n## Action Items\n")
    print(f"1. Auto-migrate {len(report.by_category['output_tensor'])} benchmarks with output tensors")
    print(f"2. Add training exemptions for {len(report.by_category['training_loop'])} training loops")
    print(f"3. Add throughput exemptions for {len(report.by_category['throughput_only'])} throughput tests")
    print(f"4. Manual CUDA handling for {len(report.by_category['cuda_binary'])} CUDA binaries")
    print(f"5. Already compliant: {len(report.by_category['already_compliant'])} benchmarks")
    
    if report.by_category['parse_error']:
        print(f"\n## Parse Errors ({len(report.by_category['parse_error'])} files)\n")
        for path in report.by_category['parse_error'][:10]:
            print(f"  - {path}")
        if len(report.by_category['parse_error']) > 10:
            print(f"  ... and {len(report.by_category['parse_error']) - 10} more")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Categorize benchmarks for migration planning"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to write JSON report"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=".",
        help="Root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    root_dir = args.root.resolve()
    
    report = categorize_all(root_dir)
    print_report(report)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nJSON report written to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


