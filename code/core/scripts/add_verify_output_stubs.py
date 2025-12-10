#!/usr/bin/env python3
"""Add get_verify_output() stubs to ALL benchmark classes.

This script scans all benchmark files and adds get_verify_output() methods
where missing. For benchmarks with detected output attributes, it generates
a working implementation. For others, it generates a NotImplementedError stub.

This enforces STRICT verification compliance - no rubber-stamping allowed.

Usage:
    python -m core.scripts.add_verify_output_stubs [--dry-run] [--chapter CHAPTER]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class BenchmarkInfo:
    """Information about a benchmark file."""
    path: Path
    benchmark_class: Optional[str] = None
    has_get_verify_output: bool = False
    output_attrs: Set[str] = field(default_factory=set)
    is_cuda_binary: bool = False
    is_training: bool = False
    is_throughput_only: bool = False
    errors: List[str] = field(default_factory=list)


def find_benchmark_files(root_dir: Path, chapter: Optional[str] = None) -> List[Path]:
    """Find all benchmark files (baseline_*.py, optimized_*.py)."""
    files: List[Path] = []
    patterns = ["baseline_*.py", "optimized_*.py"]
    
    if chapter:
        search_dirs = [root_dir / chapter]
    else:
        search_dirs = [
            d for d in root_dir.iterdir()
            if d.is_dir() and (d.name.startswith("ch") or d.name == "labs" or d.name == "llm_patches")
        ]
        # Also search labs subdirectories
        labs_dir = root_dir / "labs"
        if labs_dir.exists():
            search_dirs.extend(d for d in labs_dir.iterdir() if d.is_dir())
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            files.extend(search_dir.glob(pattern))
            files.extend(search_dir.glob(f"**/{pattern}"))
    
    return sorted(set(files))


def analyze_file(file_path: Path) -> BenchmarkInfo:
    """Analyze a benchmark file."""
    info = BenchmarkInfo(path=file_path)
    
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        info.errors.append(f"Syntax error: {e}")
        return info
    except Exception as e:
        info.errors.append(f"Parse error: {e}")
        return info
    
    # Known output attribute names
    output_attrs = {
        "output", "C", "result", "out", "y", "logits", "hidden_states",
        "output_tensor", "output_data", "result_tensor"
    }
    
    # Training indicators
    training_indicators = {"optimizer", "loss", "backward", "grad"}
    
    # Known benchmark base classes (classes inheriting from these are benchmarks)
    # Any class containing these substrings in their base class name is considered a benchmark
    benchmark_base_classes = {
        # Core
        "BaseBenchmark", "CudaBinaryBenchmark", "OccupancyBinaryBenchmark",
        "BaselineContinuousBatchingBenchmark", "BaselineDisaggregatedInferenceBenchmark",
        # ch04 bases
        "NvshmemIbgdaMicrobench",
        # ch08 bases
        "AiOptimizationBenchmarkBase", "ThresholdBenchmarkBase", "TilingBenchmarkBase",
        "HBMBenchmarkBase", "LoopUnrollingBenchmarkBase", "ThresholdBenchmarkBaseTMA",
        "TilingBenchmarkBaseTCGen05",
        # ch10 bases
        "BaselineMatmulTCGen05Benchmark",
        # ch11 bases
        "StridedStreamBaseline", "ConcurrentStreamOptimized", "StreamOrderedBase",
        # labs/train_distributed
        "TorchrunScriptBenchmark",
        # labs/moe bases
        "MoEJourneyBenchmark", "MoEBenchmarkBase", "MOETokenBenchmark",
        # General pattern - anything ending in "BenchmarkBase" or containing "Benchmark" + known parent
        "BenchmarkBase",
    }
    
    # Find benchmark class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if class has benchmark_fn OR inherits from known benchmark base
            has_benchmark_fn = any(
                isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
                for item in node.body
            )
            
            # Check base classes
            inherits_from_benchmark = False
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name and any(b in base_name for b in benchmark_base_classes):
                    inherits_from_benchmark = True
                    break
            
            if not has_benchmark_fn and not inherits_from_benchmark:
                continue
            
            info.benchmark_class = node.name
            
            # Check for CudaBinaryBenchmark (already has get_verify_output)
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name:
                    if "CudaBinary" in base_name or "OccupancyBinary" in base_name:
                        info.is_cuda_binary = True
                        # CudaBinaryBenchmark already has get_verify_output()
                        info.has_get_verify_output = True
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "get_verify_output":
                        info.has_get_verify_output = True
                    
                    # Check for training indicators
                    func_source = ast.unparse(item) if hasattr(ast, 'unparse') else ""
                    if any(ind in func_source.lower() for ind in training_indicators):
                        info.is_training = True
                
                # Check for output attributes in assignments
                if isinstance(item, ast.FunctionDef):
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute):
                                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                                        if target.attr in output_attrs:
                                            info.output_attrs.add(target.attr)
            
            # If no output attrs found, might be throughput-only
            if not info.output_attrs and not info.is_cuda_binary:
                # Check for throughput/bandwidth keywords
                source_lower = source.lower()
                if any(kw in source_lower for kw in ["throughput", "bandwidth", "tokens_per_second", "tok/s"]):
                    info.is_throughput_only = True
            
            break  # Use first benchmark class found
    
    return info


def generate_verify_output_method(info: BenchmarkInfo, indent: int = 4) -> List[str]:
    """Generate get_verify_output() method code as list of lines.
    
    Returns properly indented lines ready to be inserted into the file.
    """
    base = " " * indent
    body = " " * (indent + 4)
    
    # Priority order for output attributes
    output_priority = ["output", "C", "result", "out", "y", "logits", "hidden_states"]
    
    detected_output = None
    for attr in output_priority:
        if attr in info.output_attrs:
            detected_output = attr
            break
    
    lines = [""]  # Start with blank line
    
    if detected_output:
        lines.extend([
            f'{base}def get_verify_output(self) -> torch.Tensor:',
            f'{body}"""Return output tensor for verification comparison.',
            f'{body}',
            f'{body}MANDATORY: This method must be implemented explicitly.',
            f'{body}Auto-generated based on detected output attribute: self.{detected_output}',
            f'{body}"""',
            f'{body}if not hasattr(self, "{detected_output}") or self.{detected_output} is None:',
            f'{body}    raise RuntimeError("Output not available - run benchmark_fn() first")',
            f'{body}return self.{detected_output}',
        ])
    elif info.is_cuda_binary:
        lines.extend([
            f'{base}def get_verify_output(self) -> torch.Tensor:',
            f'{body}"""Return output tensor for verification comparison.',
            f'{body}',
            f'{body}MANDATORY: CUDA binary benchmarks must implement verification.',
            f'{body}Return checksum tensor from CUDA kernel output.',
            f'{body}"""',
            f'{body}raise NotImplementedError(',
            f'{body}    "CUDA binary benchmark must implement get_verify_output() - "',
            f'{body}    "return checksum tensor from kernel output"',
            f'{body})',
        ])
    elif info.is_training:
        lines.extend([
            f'{base}def get_verify_output(self) -> torch.Tensor:',
            f'{body}"""Return output tensor for verification comparison.',
            f'{body}',
            f'{body}MANDATORY: Training benchmarks need verification via loss/gradients.',
            f'{body}Options:',
            f'{body}1. Return final loss value as 1D tensor',
            f'{body}2. Return gradient checksum',
            f'{body}3. Return model parameter checksum after training step',
            f'{body}"""',
            f'{body}raise NotImplementedError(',
            f'{body}    "Training benchmark must implement get_verify_output() - "',
            f'{body}    "return loss/gradient/parameter checksum tensor"',
            f'{body})',
        ])
    elif info.is_throughput_only:
        lines.extend([
            f'{base}def get_verify_output(self) -> torch.Tensor:',
            f'{body}"""Return output tensor for verification comparison.',
            f'{body}',
            f'{body}MANDATORY: Throughput-only benchmarks need a checksum for verification.',
            f'{body}Compute a checksum of any internal state to verify correctness.',
            f'{body}"""',
            f'{body}raise NotImplementedError(',
            f'{body}    "Throughput benchmark must implement get_verify_output() - "',
            f'{body}    "return checksum tensor verifying internal state"',
            f'{body})',
        ])
    else:
        lines.extend([
            f'{base}def get_verify_output(self) -> torch.Tensor:',
            f'{body}"""Return output tensor for verification comparison.',
            f'{body}',
            f'{body}MANDATORY: This method must be implemented explicitly.',
            f'{body}TODO: Return the output tensor that should be compared between',
            f'{body}baseline and optimized versions.',
            f'{body}"""',
            f'{body}raise NotImplementedError(',
            f'{body}    "Benchmark must implement get_verify_output() - "',
            f'{body}    "return output tensor for verification"',
            f'{body})',
        ])
    
    lines.append("")  # End with blank line
    return lines


def add_verify_output_to_file(file_path: Path, info: BenchmarkInfo, dry_run: bool = False) -> Tuple[bool, str]:
    """Add get_verify_output() to a benchmark file."""
    
    if info.has_get_verify_output:
        return False, "already has get_verify_output()"
    
    if not info.benchmark_class:
        return False, "no benchmark class found"
    
    if info.errors:
        return False, f"parse error: {info.errors[0]}"
    
    try:
        source = file_path.read_text()
    except Exception as e:
        return False, f"read error: {e}"
    
    # Find the end of the benchmark class
    lines = source.split("\n")
    class_pattern = re.compile(rf"^class\s+{re.escape(info.benchmark_class)}\s*[\(:]")
    
    class_start = None
    class_indent = 0
    
    for i, line in enumerate(lines):
        if class_pattern.match(line.lstrip()):
            class_start = i
            class_indent = len(line) - len(line.lstrip())
            break
    
    if class_start is None:
        return False, f"could not find class {info.benchmark_class}"
    
    # Find class end (next line at same or lower indent that's not empty/comment)
    class_end = len(lines) - 1
    for i in range(class_start + 1, len(lines)):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        current_indent = len(line) - len(stripped)
        if current_indent <= class_indent and stripped and not stripped.startswith("@"):
            class_end = i - 1
            break
    
    # Ensure torch import exists
    has_torch_import = "import torch" in source
    
    if dry_run:
        action = "auto-implement" if info.output_attrs else "add stub"
        return True, f"would {action}"
    
    # Generate the method with proper indentation
    method_indent = class_indent + 4  # Methods are indented from class
    method_lines = generate_verify_output_method(info, method_indent)
    
    # Add method at end of class
    # Find last non-empty line in class
    insert_line = class_end
    while insert_line > class_start and not lines[insert_line].strip():
        insert_line -= 1
    insert_line += 1
    
    # Insert the method lines
    for i, method_line in enumerate(method_lines):
        lines.insert(insert_line + i, method_line)
    
    # Add torch import if needed (AFTER __future__ imports)
    if not has_torch_import:
        # Find the right place to add - after any __future__ imports
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("from __future__ import"):
                insert_idx = i + 1
                # Skip blank lines after __future__
                while insert_idx < len(lines) and not lines[insert_idx].strip():
                    insert_idx += 1
                break
            if line.startswith("import ") or line.startswith("from "):
                insert_idx = i
                break
        lines.insert(insert_idx, "import torch")
    
    new_source = "\n".join(lines)
    
    try:
        # Create backup
        backup_path = file_path.with_suffix(".py.bak")
        file_path.rename(backup_path)
        
        # Write new file
        file_path.write_text(new_source)
        
        action = "auto-implemented" if info.output_attrs else "added stub"
        return True, action
    except Exception as e:
        return False, f"write error: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add get_verify_output() stubs to all benchmark classes"
    )
    parser.add_argument("--chapter", "-c", help="Specific chapter to process")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview changes")
    parser.add_argument("--root", "-r", default=".", help="Root directory")
    
    args = parser.parse_args()
    root_dir = Path(args.root).resolve()
    
    files = find_benchmark_files(root_dir, args.chapter)
    print(f"Found {len(files)} benchmark files\n")
    
    results: Dict[str, List[str]] = defaultdict(list)
    
    for file_path in files:
        info = analyze_file(file_path)
        success, message = add_verify_output_to_file(file_path, info, args.dry_run)
        
        if success:
            if args.dry_run:
                results["would_modify"].append(f"{file_path.name}: {message}")
            else:
                results["modified"].append(f"{file_path.name}: {message}")
        else:
            if "already has" in message:
                results["already_done"].append(file_path.name)
            elif "no benchmark class" in message:
                results["no_class"].append(file_path.name)
            else:
                results["errors"].append(f"{file_path.name}: {message}")
    
    # Print summary
    print("=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    
    if args.dry_run:
        print(f"\nWould modify: {len(results['would_modify'])} files")
        for item in results["would_modify"][:20]:
            print(f"  - {item}")
        if len(results["would_modify"]) > 20:
            print(f"  ... and {len(results['would_modify']) - 20} more")
    else:
        print(f"\nModified: {len(results['modified'])} files")
        for item in results["modified"][:20]:
            print(f"  - {item}")
        if len(results["modified"]) > 20:
            print(f"  ... and {len(results['modified']) - 20} more")
    
    print(f"\nAlready done: {len(results['already_done'])} files")
    print(f"No benchmark class: {len(results['no_class'])} files")
    print(f"Errors: {len(results['errors'])} files")
    
    if results["errors"]:
        print("\nErrors:")
        for item in results["errors"][:10]:
            print(f"  - {item}")
    
    # Categorize no_class files by chapter
    if results["no_class"]:
        print(f"\n⚠️  {len(results['no_class'])} files have no BaseBenchmark class.")
        print("    These may use different patterns (scripts, CUDA files, etc.)")
        print("    and need manual review or different verification approach.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

