#!/usr/bin/env python3
"""Script to add get_custom_metrics() to all benchmark files.

This script:
1. Finds all baseline_*.py and optimized_*.py files in ch* and labs directories
2. Determines which metrics helper to use based on the chapter/domain
3. Adds get_custom_metrics() method if not already present

Usage:
    python core/scripts/add_custom_metrics.py --dry-run   # Preview changes
    python core/scripts/add_custom_metrics.py             # Apply changes
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Map chapters to their domain and appropriate metrics helper
CHAPTER_METRICS_MAP = {
    # Chapter 1: Performance basics - general speedup
    "ch01": ("performance", "speedup"),
    # Chapter 2: Memory transfers
    "ch02": ("memory_transfer", "transfer"),
    # Chapter 3: System configuration - general
    "ch03": ("system", "speedup"),
    # Chapter 4: Multi-GPU communication
    "ch04": ("communication", "transfer"),
    # Chapter 5: Storage I/O
    "ch05": ("storage", "transfer"),
    # Chapter 6: Kernel fundamentals - bank conflicts, divergence
    "ch06": ("kernel", "kernel"),
    # Chapter 7: Memory access patterns
    "ch07": ("memory_access", "memory"),
    # Chapter 8: Optimization techniques
    "ch08": ("optimization", "optimization"),
    # Chapter 9: Compute-bound / roofline
    "ch09": ("compute", "roofline"),
    # Chapter 10: Advanced pipelines
    "ch10": ("pipeline", "roofline"),
    # Chapter 11: CUDA streams
    "ch11": ("streams", "stream"),
    # Chapter 12: CUDA graphs
    "ch12": ("graphs", "graph"),
    # Chapter 13: PyTorch optimization / precision
    "ch13": ("precision", "precision"),
    # Chapter 14: Triton / compilers
    "ch14": ("compiler", "roofline"),
    # Chapter 15: Inference architecture
    "ch15": ("inference", "inference"),
    # Chapter 16: Production inference
    "ch16": ("inference", "inference"),
    # Chapter 17: Disaggregated inference
    "ch17": ("inference", "inference"),
    # Chapter 18: Speculative decoding
    "ch18": ("speculative", "speculative"),
    # Chapter 19: Precision / quantization
    "ch19": ("precision", "precision"),
    # Chapter 20: End-to-end
    "ch20": ("e2e", "speedup"),
}

# Lab-specific mappings
LAB_METRICS_MAP = {
    "blackwell_matmul": ("compute", "roofline"),
    "flexattention": ("attention", "roofline"),
    "kv_cache": ("inference", "inference"),
    "occupancy_tuning": ("kernel", "kernel"),
    "speculative_decode": ("speculative", "speculative"),
    "distributed_training": ("training", "speedup"),
    "moe": ("moe", "roofline"),
    "nanochat": ("inference", "inference"),
    "decode_optimization": ("inference", "inference"),
    "cudnn_sdpa": ("attention", "roofline"),
    "dynamic_router": ("inference", "inference"),
    "async_input_pipeline": ("io", "transfer"),
    "train_distributed": ("training", "speedup"),
    "persistent_decode": ("inference", "inference"),
    "real_world_models": ("training", "speedup"),
}

# Templates for different metric types
# Note: Use dict() constructor to avoid curly brace issues with .format()
METRICS_TEMPLATES = {
    "speedup": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics for performance analysis."""
        # Basic metrics - override in subclass for domain-specific values
        return dict([
            ("{prefix}.workload_size", float(getattr(self, 'batch_size', 0) or getattr(self, 'N', 0) or 0)),
        ])
''',
    "transfer": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory transfer metrics for bandwidth analysis."""
        bytes_moved = getattr(self, 'N', 0) * 4  # Estimate: elements * 4 bytes
        return dict([
            ("{prefix}.bytes_transferred", float(bytes_moved)),
            ("{prefix}.transfer_type", 0.0),  # 0=pcie, 1=nvlink, 2=hbm
        ])
''',
    "kernel": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return kernel fundamentals metrics."""
        return dict([
            ("{prefix}.elements", float(getattr(self, 'N', 0) or getattr(self, 'num_elements', 0) or 0)),
            ("{prefix}.iterations", float(getattr(self, 'repeats', 1) or getattr(self, 'iterations', 1))),
        ])
''',
    "memory": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access pattern metrics."""
        n = getattr(self, 'N', 0) or getattr(self, 'num_elements', 0) or 0
        return dict([
            ("{prefix}.elements", float(n)),
            ("{prefix}.bytes_per_element", 4.0),  # float32 default
        ])
''',
    "optimization": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization technique metrics."""
        return dict([
            ("{prefix}.technique", 0.0),  # Override with technique identifier
            ("{prefix}.registers_per_thread", 0.0),  # Fill from launch config if known
            ("{prefix}.shared_mem_bytes", 0.0),
        ])
''',
    "roofline": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return dict([
            ("{prefix}.estimated_flops", flops),
            ("{prefix}.estimated_bytes", bytes_moved),
            ("{prefix}.arithmetic_intensity", arithmetic_intensity),
        ])
''',
    "stream": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return CUDA stream metrics."""
        return dict([
            ("{prefix}.num_streams", float(getattr(self, 'num_streams', 1))),
            ("{prefix}.num_operations", float(getattr(self, 'num_operations', 1) or 3)),
            ("{prefix}.has_overlap", 0.0),  # 0=baseline (no overlap), 1=optimized
        ])
''',
    "graph": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return CUDA graph metrics."""
        return dict([
            ("{prefix}.num_iterations", float(getattr(self, 'iterations', 1) or getattr(self, 'num_iterations', 1))),
            ("{prefix}.uses_graph", 0.0),  # 0=baseline (no graph), 1=optimized
        ])
''',
    "precision": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return precision/quantization metrics."""
        return dict([
            ("{prefix}.batch_size", float(getattr(self, 'batch_size', 0) or 0)),
            ("{prefix}.hidden_dim", float(getattr(self, 'hidden_dim', 0) or 0)),
            ("{prefix}.precision_bits", 32.0),  # Override: 32=fp32, 16=fp16, 8=fp8, 4=fp4
        ])
''',
    "inference": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return dict([
            ("{prefix}.batch_size", float(getattr(self, 'batch_size', 0) or getattr(self, 'batch', 0) or 0)),
            ("{prefix}.seq_len", float(getattr(self, 'seq_len', 0) or getattr(self, 'seq_length', 0) or 0)),
            ("{prefix}.hidden_dim", float(getattr(self, 'hidden_dim', 0) or getattr(self, 'hidden', 0) or 0)),
        ])
''',
    "speculative": '''
    def get_custom_metrics(self) -> Optional[dict]:
        """Return speculative decoding metrics."""
        return dict([
            ("{prefix}.num_draft_tokens", float(getattr(self, 'num_draft_tokens', 4))),
            ("{prefix}.batch_size", float(getattr(self, 'batch_size', 1))),
        ])
''',
}


def get_chapter_from_path(filepath: Path) -> Optional[str]:
    """Extract chapter identifier from file path."""
    parts = filepath.parts
    for part in parts:
        if part.startswith("ch") and len(part) <= 4:
            return part
    return None


def get_lab_from_path(filepath: Path) -> Optional[str]:
    """Extract lab identifier from file path."""
    parts = filepath.parts
    for i, part in enumerate(parts):
        if part == "labs" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def get_metrics_type(filepath: Path) -> Tuple[str, str]:
    """Determine the appropriate metrics type for a file."""
    chapter = get_chapter_from_path(filepath)
    if chapter and chapter in CHAPTER_METRICS_MAP:
        return CHAPTER_METRICS_MAP[chapter]
    
    lab = get_lab_from_path(filepath)
    if lab:
        # Try exact match first
        if lab in LAB_METRICS_MAP:
            return LAB_METRICS_MAP[lab]
        # Try partial match
        for lab_name, metrics in LAB_METRICS_MAP.items():
            if lab_name in lab or lab in lab_name:
                return metrics
    
    return ("general", "speedup")


def has_custom_metrics(content: str) -> bool:
    """Check if file already has get_custom_metrics method."""
    return "def get_custom_metrics" in content


def has_base_benchmark(content: str) -> bool:
    """Check if file has a BaseBenchmark class."""
    return "BaseBenchmark" in content and "class " in content


def find_insertion_point(content: str) -> Optional[int]:
    """Find the best place to insert get_custom_metrics method.
    
    Looks for:
    1. After get_workload_metadata method
    2. After get_config method
    3. Before validate_result method
    4. Before teardown method
    """
    # Try to find after get_workload_metadata
    match = re.search(r'def get_workload_metadata\([^)]*\)[^:]*:.*?(?=\n    def |\nclass |\ndef [a-z]|\Z)', 
                      content, re.DOTALL)
    if match:
        return match.end()
    
    # Try to find after get_config
    match = re.search(r'def get_config\([^)]*\)[^:]*:.*?(?=\n    def |\nclass |\ndef [a-z]|\Z)', 
                      content, re.DOTALL)
    if match:
        return match.end()
    
    # Try to find before validate_result
    match = re.search(r'\n    def validate_result\(', content)
    if match:
        return match.start()
    
    # Try to find before teardown
    match = re.search(r'\n    def teardown\(', content)
    if match:
        return match.start()
    
    return None


def get_prefix_from_filename(filename: str) -> str:
    """Extract a metric prefix from the filename."""
    # Remove baseline_ or optimized_ prefix and .py suffix
    name = filename.replace("baseline_", "").replace("optimized_", "").replace(".py", "")
    # Convert to valid metric prefix (replace special chars)
    prefix = re.sub(r'[^a-z0-9]', '_', name.lower())
    return prefix[:20]  # Limit length


def add_custom_metrics_to_file(filepath: Path, dry_run: bool = True) -> bool:
    """Add get_custom_metrics method to a benchmark file.
    
    Returns True if file was modified (or would be in dry_run mode).
    """
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return False
    
    # Skip if already has get_custom_metrics
    if has_custom_metrics(content):
        return False
    
    # Skip if not a BaseBenchmark
    if not has_base_benchmark(content):
        return False
    
    # Find insertion point
    insertion_point = find_insertion_point(content)
    if insertion_point is None:
        print(f"  SKIP {filepath.name}: Could not find insertion point")
        return False
    
    # Get metrics type and template
    domain, metrics_type = get_metrics_type(filepath)
    template = METRICS_TEMPLATES.get(metrics_type, METRICS_TEMPLATES["speedup"])
    
    # Get prefix for metrics
    prefix = get_prefix_from_filename(filepath.name)
    
    # Format template with prefix
    metrics_code = template.format(prefix=prefix)
    
    # No import changes needed - we use Optional[dict] which requires no extra imports
    
    # Insert the method
    new_content = content[:insertion_point] + metrics_code + content[insertion_point:]
    
    if dry_run:
        print(f"  WOULD ADD to {filepath.name}: {domain}/{metrics_type} metrics")
        return True
    
    try:
        filepath.write_text(new_content)
        print(f"  ADDED to {filepath.name}: {domain}/{metrics_type} metrics")
        return True
    except Exception as e:
        print(f"  ERROR writing {filepath}: {e}")
        return False


def find_benchmark_files() -> List[Path]:
    """Find all benchmark files in ch* and labs directories."""
    files = []
    
    # Find in chapter directories
    for pattern in ["ch*/baseline_*.py", "ch*/optimized_*.py"]:
        files.extend(REPO_ROOT.glob(pattern))
    
    # Find in labs directories
    for pattern in ["labs/*/baseline_*.py", "labs/*/optimized_*.py"]:
        files.extend(REPO_ROOT.glob(pattern))
    
    # Filter out __pycache__ and test files
    files = [f for f in files if "__pycache__" not in str(f) and "test_" not in f.name]
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Add get_custom_metrics() to benchmark files")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    parser.add_argument("--chapter", type=str, help="Only process specific chapter (e.g., ch06)")
    parser.add_argument("--lab", type=str, help="Only process specific lab (e.g., flexattention)")
    args = parser.parse_args()
    
    files = find_benchmark_files()
    
    if args.chapter:
        files = [f for f in files if args.chapter in str(f)]
    if args.lab:
        files = [f for f in files if args.lab in str(f)]
    
    print(f"Found {len(files)} benchmark files")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")
    
    modified = 0
    skipped_existing = 0
    skipped_other = 0
    
    for filepath in files:
        content = filepath.read_text()
        if has_custom_metrics(content):
            skipped_existing += 1
            continue
        if not has_base_benchmark(content):
            skipped_other += 1
            continue
        
        if add_custom_metrics_to_file(filepath, dry_run=args.dry_run):
            modified += 1
    
    print(f"\nSummary:")
    print(f"  Modified (or would modify): {modified}")
    print(f"  Skipped (already has get_custom_metrics): {skipped_existing}")
    print(f"  Skipped (not a BaseBenchmark): {skipped_other}")
    
    if args.dry_run and modified > 0:
        print(f"\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
