"""Benchmark discovery utilities.

Provides functions to discover benchmarks across chapters and CUDA benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def discover_benchmarks(chapter_dir: Path) -> List[Tuple[Path, List[Path], str]]:
    """Discover benchmark modules by looking for baseline_*.py files with matching optimized_*.py.
    
    Args:
        chapter_dir: Path to chapter directory (e.g., Path('ch16'))
        
    Returns:
        List of tuples: (baseline_path, [optimized_paths], example_name)
        Example: (Path('ch16/baseline_moe_dense.py'), [Path('ch16/optimized_moe_sparse.py')], 'moe')
    """
    pairs = []
    baseline_files = list(chapter_dir.glob("baseline_*.py"))
    example_names = {
        baseline_file.stem.replace("baseline_", "")
        for baseline_file in baseline_files
    }
    
    for baseline_file in baseline_files:
        # Extract example name using the entire suffix after "baseline_"
        # This preserves variants like "moe_dense" instead of collapsing everything to "moe".
        example_name = baseline_file.stem.replace("baseline_", "")
        optimized_files: List[Path] = []
        
        # Pattern 1: optimized_{name}_*.py (e.g., optimized_moe_sparse.py)
        pattern1 = chapter_dir / f"optimized_{example_name}_*.py"
        for opt_path in pattern1.parent.glob(pattern1.name):
            suffix = opt_path.stem.replace(f"optimized_{example_name}_", "", 1)
            candidate_name = f"{example_name}_{suffix}"
            if candidate_name in example_names:
                continue
            optimized_files.append(opt_path)
        
        # Pattern 2: optimized_{name}.py (e.g., optimized_moe.py)
        pattern2 = chapter_dir / f"optimized_{example_name}.py"
        if pattern2.exists():
            optimized_files.append(pattern2)
        
        if optimized_files:
            pairs.append((baseline_file, optimized_files, example_name))
    
    return pairs


def discover_cuda_benchmarks(repo_root: Path) -> List[Path]:
    """Discover CUDA benchmark files (files with .cu extension or in cuda/ directories).
    
    Args:
        repo_root: Path to repository root
        
    Returns:
        List of paths to CUDA benchmark files
    """
    cuda_benchmarks = []
    
    # Look for .cu files in chapter directories
    for chapter_dir in repo_root.glob("ch*/"):
        if chapter_dir.is_dir():
            cuda_files = list(chapter_dir.glob("*.cu"))
            cuda_benchmarks.extend(cuda_files)
            
            # Also check for cuda/ subdirectories
            cuda_subdir = chapter_dir / "cuda"
            if cuda_subdir.exists() and cuda_subdir.is_dir():
                cuda_files_subdir = list(cuda_subdir.glob("*.cu"))
                cuda_benchmarks.extend(cuda_files_subdir)
    
    return sorted(cuda_benchmarks)


def discover_all_chapters(repo_root: Path) -> List[Path]:
    """Discover all chapter directories.
    
    Args:
        repo_root: Path to repository root
        
    Returns:
        List of chapter directory paths, sorted numerically (ch1, ch2, ..., ch10, ch11, ...)
    """
    def chapter_sort_key(path: Path) -> int:
        """Extract numeric part from chapter name for natural sorting."""
        if path.name.startswith('ch') and path.name[2:].isdigit():
            return int(path.name[2:])
        return 0
    
    chapter_dirs = sorted([
        d for d in repo_root.iterdir()
        if d.is_dir() and d.name.startswith('ch') and d.name[2:].isdigit()
    ], key=chapter_sort_key)
    
    capstone_dir = repo_root / "capstone"
    if capstone_dir.is_dir():
        chapter_dirs.append(capstone_dir)
    return chapter_dirs


def discover_benchmark_pairs(repo_root: Path, chapter: str = "all") -> List[Tuple[Path, List[Path], str]]:
    """Discover benchmark pairs across chapters.
    
    Args:
        repo_root: Path to repository root
        chapter: Chapter identifier ('all' or specific chapter like 'ch12' or '12')
        
    Returns:
        List of tuples: (baseline_path, [optimized_paths], example_name)
    """
    all_pairs = []
    
    if chapter == "all":
        chapter_dirs = discover_all_chapters(repo_root)
    else:
        # Normalize chapter argument
        normalized = chapter.lower()
        if normalized == "capstone":
            chapter_dir = repo_root / "capstone"
            chapter_dirs = [chapter_dir] if chapter_dir.exists() else []
        else:
            if chapter.isdigit():
                chapter = f"ch{chapter}"
            elif not chapter.startswith('ch'):
                chapter = f"ch{chapter}"
            chapter_dir = repo_root / chapter
            if chapter_dir.exists():
                chapter_dirs = [chapter_dir]
            else:
                chapter_dirs = []
    
    for chapter_dir in chapter_dirs:
        pairs = discover_benchmarks(chapter_dir)
        all_pairs.extend(pairs)
    
    return all_pairs
