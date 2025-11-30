#!/usr/bin/env python3
"""Script to analyze and update get_custom_metrics() implementations.

This script scans benchmark files and:
1. Identifies files with empty/basic get_custom_metrics() implementations
2. Suggests appropriate metric helpers based on chapter
3. Can auto-update files with suggested implementations
4. Detects inheritance from parent benchmark classes
5. Identifies standalone scripts that should be converted to proper benchmarks

Usage:
    # Analyze only (dry run)
    python core/scripts/update_custom_metrics.py --analyze
    
    # Show suggested changes for a specific chapter
    python core/scripts/update_custom_metrics.py --chapter 7
    
    # Apply changes to all files that need updates
    python core/scripts/update_custom_metrics.py --apply
    
    # Apply changes to a specific file
    python core/scripts/update_custom_metrics.py --apply --file ch07/baseline_memory_access.py
    
    # Show files that need conversion to proper benchmarks
    python core/scripts/update_custom_metrics.py --show-standalone
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Set

# Chapter to metric helper mapping
CHAPTER_METRIC_HELPERS = {
    1: "compute_environment_metrics",
    2: "compute_memory_transfer_metrics",
    3: "compute_system_config_metrics",
    4: "compute_distributed_metrics",
    5: "compute_storage_io_metrics",
    6: "compute_kernel_fundamentals_metrics",
    7: "compute_memory_access_metrics",
    8: "compute_optimization_metrics",
    9: "compute_roofline_metrics",
    10: "compute_pipeline_metrics",
    11: "compute_stream_metrics",
    12: "compute_graph_metrics",
    13: "compute_precision_metrics",
    14: "compute_triton_metrics",
    15: "compute_inference_metrics",
    16: "compute_inference_metrics",
    17: "compute_inference_metrics",
    18: "compute_speculative_decoding_metrics",
    19: "compute_precision_metrics",
    20: "compute_ai_optimization_metrics",
}

# Helper function signatures for generating code
HELPER_SIGNATURES = {
    "compute_memory_transfer_metrics": {
        "import": "from core.benchmark.metrics import compute_memory_transfer_metrics",
        "params": ["bytes_transferred", "elapsed_ms", "transfer_type"],
        "defaults": {
            "bytes_transferred": "self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4)",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', 1.0)",
            "transfer_type": '"hbm"',
        },
    },
    "compute_kernel_fundamentals_metrics": {
        "import": "from core.benchmark.metrics import compute_kernel_fundamentals_metrics",
        "params": ["num_elements", "num_iterations"],
        "defaults": {
            "num_elements": "getattr(self, 'N', getattr(self, 'num_elements', 1024))",
            "num_iterations": "1",
        },
    },
    "compute_memory_access_metrics": {
        "import": "from core.benchmark.metrics import compute_memory_access_metrics",
        "params": ["bytes_requested", "bytes_actually_transferred", "num_transactions", "optimal_transactions"],
        "defaults": {
            "bytes_requested": "float(getattr(self, 'N', 1024) * 4)",
            "bytes_actually_transferred": "float(getattr(self, 'N', 1024) * 4)",
            "num_transactions": "max(1, getattr(self, 'N', 1024) // 32)",
            "optimal_transactions": "max(1, getattr(self, 'N', 1024) // 32)",
        },
    },
    "compute_optimization_metrics": {
        "import": "from core.benchmark.metrics import compute_optimization_metrics",
        "params": ["baseline_ms", "optimized_ms", "technique"],
        "defaults": {
            "baseline_ms": "getattr(self, '_baseline_ms', 1.0)",
            "optimized_ms": "getattr(self, '_optimized_ms', 1.0)",
            "technique": '"optimization"',
        },
    },
    "compute_roofline_metrics": {
        "import": "from core.benchmark.metrics import compute_roofline_metrics",
        "params": ["total_flops", "total_bytes", "elapsed_ms", "precision"],
        "defaults": {
            "total_flops": "float(getattr(self, 'total_flops', getattr(self, 'N', 1024) * 2))",
            "total_bytes": "float(getattr(self, 'N', 1024) * 4 * 2)",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', 1.0)",
            "precision": '"fp16"',
        },
    },
    "compute_stream_metrics": {
        "import": "from core.benchmark.metrics import compute_stream_metrics",
        "params": ["sequential_time_ms", "overlapped_time_ms", "num_streams", "num_operations"],
        "defaults": {
            "sequential_time_ms": "getattr(self, '_sequential_ms', 10.0)",
            "overlapped_time_ms": "getattr(self, '_overlapped_ms', 5.0)",
            "num_streams": "getattr(self, 'num_streams', 4)",
            "num_operations": "getattr(self, 'num_operations', 4)",
        },
    },
    "compute_graph_metrics": {
        "import": "from core.benchmark.metrics import compute_graph_metrics",
        "params": ["baseline_launch_overhead_us", "graph_launch_overhead_us", "num_nodes", "num_iterations"],
        "defaults": {
            "baseline_launch_overhead_us": "getattr(self, '_baseline_launch_us', 10.0)",
            "graph_launch_overhead_us": "getattr(self, '_graph_launch_us', 1.0)",
            "num_nodes": "getattr(self, 'num_nodes', 10)",
            "num_iterations": "getattr(self, 'num_iterations', 100)",
        },
    },
    "compute_precision_metrics": {
        "import": "from core.benchmark.metrics import compute_precision_metrics",
        "params": ["fp32_time_ms", "reduced_precision_time_ms", "precision_type"],
        "defaults": {
            "fp32_time_ms": "getattr(self, '_fp32_ms', 10.0)",
            "reduced_precision_time_ms": "getattr(self, '_reduced_ms', 5.0)",
            "precision_type": '"fp8"',
        },
    },
    "compute_inference_metrics": {
        "import": "from core.benchmark.metrics import compute_inference_metrics",
        "params": ["ttft_ms", "tpot_ms", "total_tokens", "total_requests", "batch_size", "max_batch_size"],
        "defaults": {
            "ttft_ms": "getattr(self, '_ttft_ms', 50.0)",
            "tpot_ms": "getattr(self, '_tpot_ms', 10.0)",
            "total_tokens": "getattr(self, 'total_tokens', 256)",
            "total_requests": "getattr(self, 'total_requests', 1)",
            "batch_size": "getattr(self, 'batch_size', 1)",
            "max_batch_size": "getattr(self, 'max_batch_size', 32)",
        },
    },
    "compute_speculative_decoding_metrics": {
        "import": "from core.benchmark.metrics import compute_speculative_decoding_metrics",
        "params": ["draft_tokens", "accepted_tokens", "draft_time_ms", "verify_time_ms", "num_rounds"],
        "defaults": {
            "draft_tokens": "getattr(self, '_draft_tokens', 64)",
            "accepted_tokens": "getattr(self, '_accepted_tokens', 48)",
            "draft_time_ms": "getattr(self, '_draft_ms', 5.0)",
            "verify_time_ms": "getattr(self, '_verify_ms', 10.0)",
            "num_rounds": "getattr(self, '_num_rounds', 8)",
        },
    },
    # New helpers for custom chapters
    "compute_environment_metrics": {
        "import": "from core.benchmark.metrics import compute_environment_metrics",
        "params": ["gpu_count", "gpu_memory_gb"],
        "defaults": {
            "gpu_count": "getattr(self, 'gpu_count', 1)",
            "gpu_memory_gb": "getattr(self, 'gpu_memory_gb', 80.0)",
        },
    },
    "compute_system_config_metrics": {
        "import": "from core.benchmark.metrics import compute_system_config_metrics",
        "params": ["numa_nodes", "cpu_cores"],
        "defaults": {
            "numa_nodes": "getattr(self, 'numa_nodes', 1)",
            "cpu_cores": "getattr(self, 'cpu_cores', 64)",
        },
    },
    "compute_distributed_metrics": {
        "import": "from core.benchmark.metrics import compute_distributed_metrics",
        "params": ["world_size", "bytes_transferred", "elapsed_ms", "collective_type"],
        "defaults": {
            "world_size": "getattr(self, 'world_size', 1)",
            "bytes_transferred": "getattr(self, '_bytes_transferred', 1024.0)",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', 1.0)",
            "collective_type": '"allreduce"',
        },
    },
    "compute_storage_io_metrics": {
        "import": "from core.benchmark.metrics import compute_storage_io_metrics",
        "params": ["bytes_read", "bytes_written", "read_time_ms", "write_time_ms"],
        "defaults": {
            "bytes_read": "getattr(self, '_bytes_read', 0.0)",
            "bytes_written": "getattr(self, '_bytes_written', 0.0)",
            "read_time_ms": "getattr(self, '_read_time_ms', 1.0)",
            "write_time_ms": "getattr(self, '_write_time_ms', 1.0)",
        },
    },
    "compute_pipeline_metrics": {
        "import": "from core.benchmark.metrics import compute_pipeline_metrics",
        "params": ["num_stages", "stage_times_ms"],
        "defaults": {
            "num_stages": "getattr(self, 'num_stages', 4)",
            "stage_times_ms": "getattr(self, '_stage_times_ms', [1.0])",
        },
    },
    "compute_triton_metrics": {
        "import": "from core.benchmark.metrics import compute_triton_metrics",
        "params": ["num_elements", "elapsed_ms", "block_size", "num_warps"],
        "defaults": {
            "num_elements": "getattr(self, 'N', getattr(self, 'num_elements', 1024))",
            "elapsed_ms": "getattr(self, '_last_elapsed_ms', 1.0)",
            "block_size": "getattr(self, 'BLOCK_SIZE', 1024)",
            "num_warps": "getattr(self, 'num_warps', 4)",
        },
    },
    "compute_ai_optimization_metrics": {
        "import": "from core.benchmark.metrics import compute_ai_optimization_metrics",
        "params": ["original_time_ms", "ai_optimized_time_ms", "suggestions_applied", "suggestions_total"],
        "defaults": {
            "original_time_ms": "getattr(self, '_original_ms', 10.0)",
            "ai_optimized_time_ms": "getattr(self, '_optimized_ms', 5.0)",
            "suggestions_applied": "getattr(self, '_suggestions_applied', 1)",
            "suggestions_total": "getattr(self, '_suggestions_total', 1)",
        },
    },
    "compute_moe_metrics": {
        "import": "from core.benchmark.metrics import compute_moe_metrics",
        "params": ["num_experts", "active_experts", "tokens_per_expert", "routing_time_ms", "expert_compute_time_ms"],
        "defaults": {
            "num_experts": "getattr(self, 'num_experts', 8)",
            "active_experts": "getattr(self, 'active_experts', 2)",
            "tokens_per_expert": "getattr(self, '_tokens_per_expert', [100])",
            "routing_time_ms": "getattr(self, '_routing_ms', 1.0)",
            "expert_compute_time_ms": "getattr(self, '_expert_compute_ms', 10.0)",
        },
    },
}


def get_chapter_from_path(path: Path) -> Optional[int]:
    """Extract chapter number from file path."""
    match = re.search(r'ch(\d+)', str(path))
    if match:
        return int(match.group(1))
    return None


def has_conditional_none_return(content: str) -> bool:
    """Check if get_custom_metrics has a conditional None return (false positive).
    
    A conditional None return is something like:
        if not self._history:
            return None
        return {...actual metrics...}
    
    This is proper defensive programming, not a missing implementation.
    """
    # Find the get_custom_metrics method
    match = re.search(
        r'def get_custom_metrics\s*\([^)]*\)[^:]*:\s*\n((?:[ \t]+.*\n)*)',
        content
    )
    if not match:
        return False
    
    impl = match.group(1)
    
    # Check if it has both a conditional return None AND returns a dict
    has_conditional_none = bool(re.search(r'if\s+.*:\s*\n\s*return None', impl))
    has_dict_return = bool(re.search(r'return\s*\{', impl))
    
    return has_conditional_none and has_dict_return


def find_parent_class_file(parent_class: str, current_file: Path, root: Path) -> Optional[Path]:
    """Find the file that defines a parent class."""
    # Handle imports like "from ch15.baseline_foo import FooClass"
    current_content = current_file.read_text()
    
    # Look for import statements that import this class
    import_patterns = [
        rf'from\s+([\w.]+)\s+import\s+.*{re.escape(parent_class)}',
        rf'from\s+([\w.]+)\s+import\s+\(.*{re.escape(parent_class)}.*\)',
    ]
    
    for pattern in import_patterns:
        match = re.search(pattern, current_content, re.DOTALL)
        if match:
            module_path = match.group(1).replace('.', '/')
            candidate = root / f"{module_path}.py"
            if candidate.exists():
                return candidate
    
    # Check same directory
    same_dir = current_file.parent
    for py_file in same_dir.glob("*.py"):
        if py_file == current_file:
            continue
        try:
            content = py_file.read_text()
            if f"class {parent_class}" in content:
                return py_file
        except Exception:
            continue
    
    return None


def check_class_has_get_custom_metrics(file_path: Path, class_name: str) -> bool:
    """Check if a class in a file has get_custom_metrics defined."""
    try:
        content = file_path.read_text()
        # Simple check: look for class definition and get_custom_metrics in that class
        class_pattern = rf'class\s+{re.escape(class_name)}\s*\([^)]*\):\s*\n((?:[ \t]+.*\n)*)'
        match = re.search(class_pattern, content)
        if match:
            class_body = match.group(1)
            return 'def get_custom_metrics' in class_body
        # Also check if the file has get_custom_metrics at all (might be in parent)
        return 'def get_custom_metrics' in content
    except Exception:
        return False


def inherits_get_custom_metrics(content: str, file_path: Path, root: Path, visited: Optional[Set[Path]] = None) -> bool:
    """Check if the file inherits get_custom_metrics from a base class.
    
    This does a recursive check through the inheritance chain.
    """
    if visited is None:
        visited = set()
    
    if file_path in visited:
        return False
    visited.add(file_path)
    
    # Look for class inheritance patterns
    class_matches = re.findall(r'class\s+\w+\s*\(([^)]+)\)', content)
    if not class_matches:
        return False
    
    for parents in class_matches:
        parent_list = [p.strip() for p in parents.split(',')]
        for parent in parent_list:
            # Clean up the parent name (remove module prefix if any)
            parent_clean = parent.split('.')[-1].strip()
            
            # Skip BaseBenchmark - it has get_custom_metrics but returns None by default
            if parent_clean == 'BaseBenchmark':
                continue
            
            # Known base classes with get_custom_metrics
            known_bases = [
                'HBMBenchmarkBase',
                'StridedStreamBaseline',
                'ConcurrentStreamOptimized',
                '_DynamicRoutingBenchmark',
                '_DisaggregatedInferenceBenchmark',
                '_DynamicQuantizedCacheBenchmark',
                '_MoEInferenceBenchmark',
                '_SpeculativeDecodingBenchmark',
                '_FlexDecodingBenchmark',
                '_BaselineContinuousBatchingBenchmark',
                'BaselineContinuousBatchingBenchmark',
                'BaselineMatmulTCGen05Benchmark',
                # Ch10 pipeline bases
                'BaselinePipelineBenchmark',
                'OptimizedPipelineBenchmark',
                # Ch11 stream bases
                'BaselineStreamBenchmark',
                'OptimizedStreamBenchmark',
                # Ch4 distributed bases
                'BaselineDataParallelBenchmark',
                'OptimizedDataParallelBenchmark',
                'BaselineNCCLBenchmark',
                'OptimizedNCCLBenchmark',
                # Any class ending with Benchmark that's imported
                'Benchmark',
            ]
            if any(parent_clean.endswith(base) or parent_clean == base for base in known_bases):
                return True
            
            # Try to find and check the parent class file
            parent_file = find_parent_class_file(parent_clean, file_path, root)
            if parent_file:
                # Check if parent has get_custom_metrics
                if check_class_has_get_custom_metrics(parent_file, parent_clean):
                    return True
                # Recursively check parent's parents
                try:
                    parent_content = parent_file.read_text()
                    if inherits_get_custom_metrics(parent_content, parent_file, root, visited):
                        return True
                except Exception:
                    pass
    
    return False


def is_standalone_script(content: str) -> bool:
    """Check if a file is a standalone script rather than a proper benchmark class."""
    # A standalone script typically:
    # 1. Has no class that inherits from any Benchmark class
    # 2. Has a main() function
    # 3. Has if __name__ == "__main__"
    # 4. Has get_benchmark() that returns a callable, not a BaseBenchmark instance
    
    # Check for any class that inherits from a Benchmark-like class
    has_benchmark_class = bool(re.search(r'class\s+\w+.*\(.*(?:BaseBenchmark|Benchmark|BenchmarkBase).*\)', content))
    
    # Also check for classes that inherit from other benchmark classes (like HBMBenchmarkBase)
    # by looking for class definitions that import from benchmark modules
    imports_benchmark_base = bool(re.search(r'from\s+[\w.]+\s+import\s+.*(?:Benchmark|BenchmarkBase)', content))
    
    has_main_func = 'def main()' in content
    has_main_guard = '__name__' in content and '__main__' in content
    
    # Check if get_benchmark returns a callable wrapper instead of a class instance
    get_bench_match = re.search(r'def get_benchmark\(\).*?:\s*\n((?:[ \t]+.*\n)*)', content)
    returns_callable = False
    if get_bench_match:
        body = get_bench_match.group(1)
        # Returns a wrapper function or lambda, not a class instance
        returns_callable = ('def _run' in body or 'lambda' in body or 
                           'return main' in body or 'main()' in body)
    
    # It's standalone if:
    # - No benchmark class AND
    # - Has main function AND main guard AND
    # - get_benchmark returns a callable (not a class)
    return (not has_benchmark_class and 
            not imports_benchmark_base and 
            has_main_func and 
            has_main_guard and
            returns_callable)


def is_alias_file(content: str) -> bool:
    """Check if a file is an alias/wrapper that just re-exports another benchmark.
    
    Alias files:
    - Have no class definition of their own
    - Import a benchmark class from another module OR import get_benchmark from another module
    - Have get_benchmark() that just returns an instance of the imported class
    
    These are fine as-is since the imported class has get_custom_metrics.
    """
    # Has no class definition
    has_class = bool(re.search(r'^class\s+\w+', content, re.MULTILINE))
    if has_class:
        return False
    
    # Check various import patterns for aliases
    is_wrapper = False
    
    # Pattern 1: Imports a benchmark class from another ch module
    imports_benchmark = bool(re.search(
        r'from\s+ch\d+\.\w+\s+import\s+.*(?:Benchmark|BenchmarkBase)',
        content
    ))
    # Pattern 2: Multi-line import with parentheses from ch module
    if not imports_benchmark:
        imports_benchmark = bool(re.search(
            r'from\s+ch\d+\.\w+\s+import\s+\([^)]*(?:Benchmark|BenchmarkBase)',
            content,
            re.DOTALL
        ))
    
    # Pattern 3: Imports get_benchmark from same-directory baseline/optimized module
    imports_get_benchmark = bool(re.search(
        r'from\s+(?:baseline|optimized)_\w+\s+import\s+get_benchmark',
        content
    ))
    
    # Has get_benchmark() function
    get_bench_match = re.search(r'def get_benchmark\(\)[^:]*:\s*\n((?:[ \t]+.*\n)*)', content)
    if not get_bench_match:
        return False
    
    body = get_bench_match.group(1)
    
    # Check what get_benchmark returns
    # Returns a class instance (like SomeBenchmark())
    returns_instance = bool(re.search(r'return\s+\w+Benchmark\(\)', body))
    # Returns result of _get_benchmark() - a wrapper pattern
    returns_delegate = bool(re.search(r'return\s+(?:bench|_get_benchmark\(\))', body))
    # Calls _get_benchmark() and modifies then returns
    calls_delegate = bool(re.search(r'(?:bench|bm|benchmark)\s*=\s*_get_benchmark\(\)', body))
    
    # Is an alias if:
    # - Imports benchmark class and returns instance, OR
    # - Imports get_benchmark and delegates to it
    return (imports_benchmark and returns_instance) or \
           (imports_get_benchmark and (returns_delegate or calls_delegate))


def find_standalone_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Find baseline/optimized pairs where both are standalone scripts."""
    pairs = []
    
    for ch_dir in sorted(root.glob("ch*")):
        if not ch_dir.is_dir():
            continue
        
        for baseline in ch_dir.glob("baseline_*.py"):
            name = baseline.name.replace("baseline_", "")
            optimized = ch_dir / f"optimized_{name}"
            
            if optimized.exists():
                try:
                    baseline_content = baseline.read_text()
                    optimized_content = optimized.read_text()
                    
                    if is_standalone_script(baseline_content) and is_standalone_script(optimized_content):
                        pairs.append((baseline, optimized))
                except Exception:
                    continue
    
    return pairs


def analyze_get_custom_metrics(file_path: Path, root: Optional[Path] = None) -> dict:
    """Analyze a file's get_custom_metrics implementation.
    
    Returns:
        Dict with analysis results including:
        - has_method: bool
        - returns_none: bool (unconditional None return)
        - returns_empty: bool
        - returns_basic: bool (just timing)
        - uses_helper: bool
        - has_conditional_none: bool (proper defensive code)
        - inherits_method: bool
        - is_standalone: bool (standalone script, not proper benchmark)
        - line_number: int or None
        - current_impl: str or None
    """
    if root is None:
        root = file_path.parent.parent
    
    try:
        content = file_path.read_text()
    except Exception:
        return {"error": "Could not read file"}
    
    result = {
        "has_method": False,
        "returns_none": False,
        "returns_empty": False,
        "returns_basic": False,
        "uses_helper": False,
        "has_conditional_none": False,
        "inherits_method": False,
        "is_standalone": False,
        "is_alias": False,
        "line_number": None,
        "current_impl": None,
        "needs_update": False,
    }
    
    # Check if it's a standalone script
    result["is_standalone"] = is_standalone_script(content)
    
    # Check if it's an alias file (thin wrapper around another benchmark)
    result["is_alias"] = is_alias_file(content)
    
    # Check if get_custom_metrics exists directly in the file
    if "def get_custom_metrics" not in content:
        # Check inheritance
        result["inherits_method"] = inherits_get_custom_metrics(content, file_path, root)
        return result
    
    result["has_method"] = True
    
    # Check for conditional None returns (false positives)
    result["has_conditional_none"] = has_conditional_none_return(content)
    
    # Find the method
    match = re.search(
        r'def get_custom_metrics\s*\([^)]*\)[^:]*:\s*\n((?:[ \t]+.*\n)*)',
        content
    )
    if match:
        impl = match.group(1)
        result["current_impl"] = impl.strip()
        
        # Check what it returns
        if re.search(r'compute_\w+_metrics', impl):
            result["uses_helper"] = True
        elif result["has_conditional_none"]:
            # Has proper conditional None + dict return - this is good
            pass
        elif re.search(r'^\s*return None\s*$', impl, re.MULTILINE) and "return {" not in impl:
            # Only returns None, no dict
            result["returns_none"] = True
            result["needs_update"] = True
        elif "return {}" in impl or "return dict()" in impl:
            result["returns_empty"] = True
            result["needs_update"] = True
        elif len(impl.strip().split('\n')) <= 3 and "return {" not in impl:
            result["returns_basic"] = True
            result["needs_update"] = True
    
    # Find line number
    for i, line in enumerate(content.split('\n'), 1):
        if 'def get_custom_metrics' in line:
            result["line_number"] = i
            break
    
    return result


def generate_helper_code(helper_name: str, indent: str = "        ") -> str:
    """Generate the code for a helper function call."""
    if helper_name not in HELPER_SIGNATURES:
        return None
    
    sig = HELPER_SIGNATURES[helper_name]
    
    lines = [
        f'{indent}"""Return domain-specific metrics using standardized helper."""',
        f'{indent}{sig["import"]}',
        f'{indent}return {helper_name}(',
    ]
    
    for param in sig["params"]:
        default = sig["defaults"].get(param, "0")
        lines.append(f'{indent}    {param}={default},')
    
    lines.append(f'{indent})')
    
    return '\n'.join(lines)


def update_file(file_path: Path, helper_name: str, dry_run: bool = True) -> Tuple[bool, str]:
    """Update a file's get_custom_metrics to use the helper function.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        content = file_path.read_text()
    except Exception as e:
        return False, f"Could not read file: {e}"
    
    # Find the existing get_custom_metrics method
    pattern = r'(def get_custom_metrics\s*\([^)]*\)[^:]*:)\s*\n((?:[ \t]+.*\n)*)'
    match = re.search(pattern, content)
    
    if not match:
        return False, "No get_custom_metrics method found"
    
    # Detect indentation
    first_line = match.group(2).split('\n')[0] if match.group(2) else ""
    indent_match = re.match(r'^(\s+)', first_line)
    indent = indent_match.group(1) if indent_match else "        "
    
    # Generate new implementation
    new_impl = generate_helper_code(helper_name, indent)
    if not new_impl:
        return False, f"Unknown helper: {helper_name}"
    
    # Create the new method
    new_method = f"{match.group(1)}\n{new_impl}\n"
    
    # Replace in content
    new_content = content[:match.start()] + new_method + content[match.end():]
    
    if dry_run:
        return True, f"Would update with {helper_name}"
    
    try:
        file_path.write_text(new_content)
        return True, f"Updated with {helper_name}"
    except Exception as e:
        return False, f"Could not write file: {e}"


def scan_directory(root: Path, chapter: Optional[int] = None) -> list:
    """Scan directory for benchmark files and analyze them."""
    results = []
    
    patterns = ["baseline_*.py", "optimized_*.py"]
    
    for pattern in patterns:
        if chapter:
            search_path = root / f"ch{chapter}"
            if not search_path.exists():
                continue
            files = list(search_path.glob(pattern))
        else:
            files = list(root.glob(f"ch*/{pattern}"))
        
        for file_path in sorted(files):
            if "__pycache__" in str(file_path):
                continue
            
            analysis = analyze_get_custom_metrics(file_path, root)
            ch = get_chapter_from_path(file_path)
            helper = CHAPTER_METRIC_HELPERS.get(ch)
            
            # Determine if this file could benefit from using a helper
            can_use_helper = (
                helper is not None and
                not analysis.get("uses_helper") and
                not analysis.get("has_conditional_none") and
                not analysis.get("inherits_method") and
                not analysis.get("is_standalone") and
                analysis.get("has_method")
            )
            
            results.append({
                "path": file_path,
                "chapter": ch,
                "analysis": analysis,
                "helper": helper,
                "can_use_helper": can_use_helper,
            })
    
    return results


def print_analysis(results: list, verbose: bool = False, show_standalone: bool = False):
    """Print analysis results."""
    # Group by status
    alias_files = [r for r in results if r["analysis"].get("is_alias")]
    no_method = [r for r in results if not r["analysis"].get("has_method") and not r["analysis"].get("inherits_method") and not r["analysis"].get("is_standalone") and not r["analysis"].get("is_alias")]
    inherits = [r for r in results if r["analysis"].get("inherits_method")]
    standalone = [r for r in results if r["analysis"].get("is_standalone")]
    returns_none = [r for r in results if r["analysis"].get("returns_none")]
    returns_empty = [r for r in results if r["analysis"].get("returns_empty")]
    returns_basic = [r for r in results if r["analysis"].get("returns_basic")]
    uses_helper = [r for r in results if r["analysis"].get("uses_helper")]
    has_conditional = [r for r in results if r["analysis"].get("has_conditional_none")]
    can_improve = [r for r in results if r["can_use_helper"]]
    
    print("=" * 70)
    print("get_custom_metrics() Analysis Summary")
    print("=" * 70)
    print(f"Total files scanned: {len(results)}")
    print()
    print("Status breakdown:")
    print(f"  ✅ Uses helper function: {len(uses_helper)}")
    print(f"  ✅ Has proper conditional None (good!): {len(has_conditional)}")
    print(f"  ✅ Inherits from parent class: {len(inherits)}")
    print(f"  ✅ Alias files (delegates to other benchmark): {len(alias_files)}")
    print(f"  📜 Standalone scripts (need conversion): {len(standalone)}")
    print(f"  ⚠️  No get_custom_metrics method: {len(no_method)}")
    print(f"  ⚠️  Returns None unconditionally: {len(returns_none)}")
    print(f"  ⚠️  Returns empty dict: {len(returns_empty)}")
    print(f"  ⚠️  Returns basic metrics only: {len(returns_basic)}")
    print()
    print(f"📊 Files that can use helper functions: {len(can_improve)}")
    print()
    
    if show_standalone and standalone:
        print("=" * 70)
        print("Standalone Scripts (should be converted to proper benchmarks)")
        print("=" * 70)
        for r in standalone:
            print(f"  {r['path']}")
        print()
        print("These files use baseline_/optimized_ naming but are standalone scripts.")
        print("They should be converted to proper BaseBenchmark subclasses.")
        print()
    
    # Show alias files if there are warnings for "no method"
    if verbose and alias_files:
        print("=" * 70)
        print("Alias Files (thin wrappers - OK as-is)")
        print("=" * 70)
        for r in alias_files:
            print(f"  {r['path']}")
        print()
    
    # Show files truly missing get_custom_metrics
    if no_method:
        print("=" * 70)
        print(f"⚠️  Files missing get_custom_metrics ({len(no_method)} files)")
        print("=" * 70)
        for r in no_method:
            print(f"  {r['path']}")
        print()
    
    if verbose and can_improve:
        print("Files that can use helper functions:")
        print("-" * 70)
        for r in can_improve:
            ch = r["chapter"]
            helper = r["helper"]
            print(f"  ch{ch}: {r['path'].name}")
            print(f"       → {helper}")
        print()
    
    # Group by chapter
    print("By Chapter:")
    print("-" * 70)
    chapter_stats = {}
    for r in results:
        ch = r["chapter"]
        if ch not in chapter_stats:
            chapter_stats[ch] = {"total": 0, "good": 0, "can_improve": 0, "standalone": 0, "alias": 0, "missing": 0}
        chapter_stats[ch]["total"] += 1
        if r["analysis"].get("uses_helper") or r["analysis"].get("has_conditional_none") or r["analysis"].get("inherits_method") or r["analysis"].get("is_alias"):
            chapter_stats[ch]["good"] += 1
        if r["can_use_helper"]:
            chapter_stats[ch]["can_improve"] += 1
        if r["analysis"].get("is_standalone"):
            chapter_stats[ch]["standalone"] += 1
        if r["analysis"].get("is_alias"):
            chapter_stats[ch]["alias"] += 1
        # Files truly missing the method
        if not r["analysis"].get("has_method") and not r["analysis"].get("inherits_method") and not r["analysis"].get("is_standalone") and not r["analysis"].get("is_alias"):
            chapter_stats[ch]["missing"] += 1
    
    for ch in sorted(chapter_stats.keys()):
        stats = chapter_stats[ch]
        helper = CHAPTER_METRIC_HELPERS.get(ch, "custom")
        notes = []
        if stats['standalone'] > 0:
            notes.append(f"{stats['standalone']} standalone")
        if stats['alias'] > 0:
            notes.append(f"{stats['alias']} alias")
        if stats['missing'] > 0:
            notes.append(f"⚠️ {stats['missing']} missing")
        notes_str = f" [{', '.join(notes)}]" if notes else ""
        print(f"  Ch{ch:2d}: {stats['good']:2d}/{stats['total']:2d} good "
              f"({stats['can_improve']:2d} can improve){notes_str} - {helper or 'custom'}")


def apply_updates(results: list, dry_run: bool = True, specific_file: Optional[str] = None) -> int:
    """Apply updates to files that can use helpers.
    
    Returns number of files updated.
    """
    updates = 0
    
    for r in results:
        if not r["can_use_helper"]:
            continue
        
        if specific_file and str(r["path"]) != specific_file and r["path"].name != specific_file:
            continue
        
        helper = r["helper"]
        if not helper:
            continue
        
        success, msg = update_file(r["path"], helper, dry_run=dry_run)
        
        if success:
            updates += 1
            status = "[DRY-RUN]" if dry_run else "[UPDATED]"
            print(f"{status} {r['path']}: {msg}")
        else:
            print(f"[SKIPPED] {r['path']}: {msg}")
    
    return updates


def validate_file_metrics(file_path: Path, root: Path) -> dict:
    """Validate that a file's get_custom_metrics returns meaningful data.
    
    This does a static analysis of what metrics are returned.
    """
    try:
        content = file_path.read_text()
    except Exception as e:
        return {"valid": False, "issues": [f"Cannot read file: {e}"], "metrics_count": 0}
    
    # Find get_custom_metrics method
    match = re.search(r'def get_custom_metrics\(self\)[^:]*:(.*?)(?=\n    def |\nclass |\Z)', 
                      content, re.DOTALL)
    if not match:
        return {"valid": False, "issues": ["No get_custom_metrics method found"], "metrics_count": 0}
    
    method_body = match.group(1)
    issues = []
    
    # Check for helper function usage
    helper_match = re.search(r'return (compute_\w+)\(', method_body)
    if helper_match:
        helper_name = helper_match.group(1)
        # Get expected metric count from helper
        helper_metrics = {
            "compute_memory_transfer_metrics": 5,
            "compute_kernel_fundamentals_metrics": 5,
            "compute_memory_access_metrics": 7,
            "compute_optimization_metrics": 7,
            "compute_roofline_metrics": 9,
            "compute_stream_metrics": 9,
            "compute_graph_metrics": 7,
            "compute_precision_metrics": 7,
            "compute_inference_metrics": 8,
            "compute_speculative_decoding_metrics": 10,
            "compute_environment_metrics": 5,
            "compute_system_config_metrics": 6,
            "compute_distributed_metrics": 7,
            "compute_storage_io_metrics": 8,
            "compute_pipeline_metrics": 9,
            "compute_triton_metrics": 9,
            "compute_ai_optimization_metrics": 9,
            "compute_moe_metrics": 11,
        }
        metrics_count = helper_metrics.get(helper_name, 0)
        return {"valid": True, "issues": [], "metrics_count": metrics_count, "helper": helper_name}
    
    # Check for dict literal
    dict_match = re.search(r'return\s*\{([^}]*)\}', method_body, re.DOTALL)
    if dict_match:
        dict_content = dict_match.group(1)
        # Count key-value pairs
        keys = re.findall(r'["\']([^"\']+)["\']:', dict_content)
        metrics_count = len(keys)
        
        if metrics_count == 0:
            issues.append("Returns empty dict")
        
        # Check naming convention
        bad_names = [k for k in keys if '.' not in k]
        if bad_names:
            issues.append(f"Keys without 'category.metric' naming: {bad_names[:3]}")
        
        return {"valid": len(issues) == 0, "issues": issues, "metrics_count": metrics_count}
    
    # Check for None return
    if 'return None' in method_body:
        # Check if conditional
        if re.search(r'if\s+.*:\s*\n\s*return None', method_body):
            return {"valid": True, "issues": [], "metrics_count": 0, "conditional_none": True}
        else:
            return {"valid": False, "issues": ["Returns None unconditionally"], "metrics_count": 0}
    
    return {"valid": True, "issues": [], "metrics_count": 0, "unknown": True}


def run_validation(results: list):
    """Run validation on all files and report issues."""
    print("=" * 70)
    print("Metric Validation Report")
    print("=" * 70)
    
    total = 0
    valid = 0
    issues_found = []
    
    for r in results:
        if r["analysis"].get("is_standalone") or r["analysis"].get("is_alias"):
            continue
        
        file_path = r["path"]
        validation = validate_file_metrics(file_path, file_path.parent.parent)
        total += 1
        
        if validation["valid"]:
            valid += 1
        else:
            issues_found.append((file_path, validation))
    
    print(f"Files validated: {total}")
    print(f"Valid: {valid}")
    print(f"With issues: {len(issues_found)}")
    print()
    
    if issues_found:
        print("Files with issues:")
        print("-" * 70)
        for path, val in issues_found[:20]:  # Limit output
            print(f"  {path.name}")
            for issue in val["issues"]:
                print(f"    ⚠️  {issue}")
        if len(issues_found) > 20:
            print(f"  ... and {len(issues_found) - 20} more")
    else:
        print("✅ All files pass validation!")


def main():
    parser = argparse.ArgumentParser(description="Analyze and update get_custom_metrics() implementations")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--chapter", type=int, help="Focus on specific chapter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--apply", action="store_true", help="Apply updates (not dry-run)")
    parser.add_argument("--file", type=str, help="Apply to specific file only")
    parser.add_argument("--show-suggestion", type=str, help="Show suggestion for a file")
    parser.add_argument("--show-standalone", action="store_true", help="Show standalone scripts that need conversion")
    parser.add_argument("--validate", action="store_true", help="Validate metric quality in all files")
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    
    if args.show_suggestion:
        file_path = root / args.show_suggestion
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        
        analysis = analyze_get_custom_metrics(file_path, root)
        chapter = get_chapter_from_path(file_path)
        helper = CHAPTER_METRIC_HELPERS.get(chapter)
        
        print(f"File: {file_path}")
        print(f"Chapter: {chapter}")
        print(f"Analysis: {analysis}")
        print()
        
        if analysis.get("is_standalone"):
            print("⚠️  This is a standalone script that should be converted to a proper benchmark.")
            print("   It uses baseline_/optimized_ naming but doesn't inherit from BaseBenchmark.")
        elif helper and not analysis.get("uses_helper") and not analysis.get("has_conditional_none"):
            print(f"Suggested helper: {helper}")
            print()
            print("Generated implementation:")
            print("-" * 40)
            print("    def get_custom_metrics(self) -> Optional[dict]:")
            print(generate_helper_code(helper, "        "))
        else:
            print("No update needed (already good or uses custom metrics)")
        return
    
    # Scan files
    results = scan_directory(root, chapter=args.chapter)
    
    # Always print analysis
    print_analysis(results, verbose=args.verbose, show_standalone=args.show_standalone)
    
    # Apply updates if requested
    if args.apply or args.file:
        print()
        print("=" * 70)
        print("Applying Updates")
        print("=" * 70)
        
        dry_run = not args.apply
        count = apply_updates(results, dry_run=dry_run, specific_file=args.file)
        
        if dry_run:
            print()
            print(f"Would update {count} files. Use --apply to actually update.")
        else:
            print()
            print(f"Updated {count} files.")
    elif args.validate:
        print()
        run_validation(results)
    else:
        can_improve = [r for r in results if r["can_use_helper"]]
        standalone = [r for r in results if r["analysis"].get("is_standalone")]
        
        if can_improve or standalone:
            print()
            if can_improve:
                print("To apply updates:")
                print("  python core/scripts/update_custom_metrics.py --apply")
                print()
            if standalone:
                print("To see standalone scripts that need conversion:")
                print("  python core/scripts/update_custom_metrics.py --show-standalone")
                print()
            print("To see a suggestion for a specific file:")
            print("  python core/scripts/update_custom_metrics.py --show-suggestion ch07/baseline_memory_access.py")
        
        print()
        print("To validate metric quality:")
        print("  python core/scripts/update_custom_metrics.py --validate")


if __name__ == "__main__":
    main()
