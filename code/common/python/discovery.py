"""Benchmark discovery utilities.

Provides functions to discover benchmarks across chapters and CUDA benchmarks.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _has_get_benchmark(file_path: Path) -> bool:
    """Quick check if a Python file has get_benchmark() function.
    
    Does a simple text search without importing the module.
    """
    try:
        source = file_path.read_text()
        return "def get_benchmark" in source
    except Exception:
        return False


def validate_benchmark_file(file_path: Path, warn: bool = True) -> bool:
    """Validate that a benchmark file has get_benchmark().
    
    Args:
        file_path: Path to benchmark file
        warn: If True, emit a warning for missing get_benchmark()
        
    Returns:
        True if file has get_benchmark(), False otherwise
    """
    if not file_path.suffix == ".py":
        return True  # Skip non-Python files
    
    has_fn = _has_get_benchmark(file_path)
    
    if not has_fn and warn:
        warnings.warn(
            f"Benchmark file '{file_path.name}' is missing get_benchmark() function. "
            f"Add: def get_benchmark() -> BaseBenchmark: return YourClass()",
            UserWarning,
            stacklevel=2
        )
    
    return has_fn

# LAB_NAMES is deprecated - labs are now auto-discovered based on
# presence of baseline_*.py or level*.py files in labs/ subdirectories.
# Kept for backward compatibility but no longer required for registration.
LAB_NAMES = set()  # Auto-discovery handles this now
# Shorthand aliases for common labs (optional convenience)
LAB_ALIASES: Dict[str, str] = {
    "capstone": "labs/fullstack_cluster",
    "moe_journey": "labs/moe_optimization_journey",
}


def _labs_root(repo_root: Path) -> Path:
    return repo_root / "labs"


def _lab_dirs(repo_root: Path) -> Iterable[Path]:
    """Auto-discover all lab directories that contain benchmark files."""
    labs_root = _labs_root(repo_root)
    if not labs_root.is_dir():
        return []
    
    lab_dirs = []
    for p in labs_root.iterdir():
        if not p.is_dir() or p.name.startswith(('_', '.')):
            continue
        # Check if directory has any baseline_*.py files (benchmark convention)
        has_benchmarks = any(p.glob("baseline_*.py")) or any(p.glob("level*.py"))
        if has_benchmarks:
            lab_dirs.append(p)
    
    return sorted(lab_dirs, key=lambda p: p.name)


def chapter_slug(chapter_dir: Path, repo_root: Path) -> str:
    """Return a consistent chapter identifier relative to repo_root."""
    return str(chapter_dir.resolve().relative_to(repo_root).as_posix())


def normalize_chapter_token(token: str) -> str:
    """Normalize chapter token (CLI arg or alias) to a relative path slug.

    Examples:
      - 'ch10' -> 'ch10'
      - '10' -> 'ch10'
      - 'labs/blackwell_matmul' -> 'labs/blackwell_matmul'
      - 'lab_blackwell_matmul' -> 'labs/blackwell_matmul'
      - 'blackwell_matmul' -> 'labs/blackwell_matmul'
      - 'capstone2' -> 'labs/blackwell_matmul'
    """
    chapter = token.strip().lower()
    if not chapter:
        raise ValueError("Chapter token cannot be empty.")

    if chapter.isdigit():
        return f"ch{chapter}"

    chapter = chapter.replace("labs.", "labs/").replace("\\", "/")

    if chapter in LAB_ALIASES:
        return LAB_ALIASES[chapter]

    if chapter.startswith("lab_"):
        trimmed = chapter[len("lab_") :]
        if trimmed in LAB_NAMES:
            return f"labs/{trimmed}"

    if chapter.startswith("labs/"):
        _, _, suffix = chapter.partition("/")
        if suffix in LAB_NAMES:
            return chapter

    if chapter in LAB_NAMES:
        return f"labs/{chapter}"

    if chapter.startswith("ch") and chapter[2:].isdigit():
        return chapter

    raise ValueError(
        f"Invalid chapter identifier '{token}'. Expected formats like "
        "'ch3', 'labs/blackwell_matmul', or 'blackwell_matmul'."
    )


def discover_benchmarks(
    chapter_dir: Path, 
    validate: bool = False,
    warn_missing: bool = False
) -> List[Tuple[Path, List[Path], str]]:
    """Discover benchmark modules by looking for baseline_*.py files with matching optimized_*.py.
    
    Args:
        chapter_dir: Path to chapter directory (e.g., Path('ch16'))
        validate: If True, check that files have get_benchmark() and skip those that don't
        warn_missing: If True, emit warnings for files missing get_benchmark()
        
    Returns:
        List of tuples: (baseline_path, [optimized_paths], example_name)
        Example: (Path('ch16/baseline_moe_dense.py'), [Path('ch16/optimized_moe_sparse.py')], 'moe')
    """
    pairs = []
    baseline_files = sorted(chapter_dir.glob("baseline_*.py"))

    example_names = {
        baseline_file.stem.replace("baseline_", "")
        for baseline_file in baseline_files
    }
    
    for baseline_file in baseline_files:
        # Validate baseline file if requested
        if validate or warn_missing:
            baseline_valid = validate_benchmark_file(baseline_file, warn=warn_missing)
            if validate and not baseline_valid:
                continue  # Skip invalid baseline files
        
        # Extract example name using the entire suffix after "baseline_"
        # This preserves variants like "moe_dense" instead of collapsing everything to "moe".
        example_name = baseline_file.stem.replace("baseline_", "")
        optimized_files: List[Path] = []
        variant_aliases: List[Tuple[str, Path]] = []
        ext = baseline_file.suffix or ".py"
        
        # Pattern 1: optimized_{name}_*.{ext} (e.g., optimized_moe_sparse.py)
        pattern1 = chapter_dir / f"optimized_{example_name}_*{ext}"
        for opt_path in pattern1.parent.glob(pattern1.name):
            # Validate optimized file if requested
            if validate or warn_missing:
                opt_valid = validate_benchmark_file(opt_path, warn=warn_missing)
                if validate and not opt_valid:
                    continue  # Skip invalid optimized files
            
            suffix = opt_path.stem.replace(f"optimized_{example_name}_", "", 1)
            candidate_name = f"{example_name}_{suffix}"
            if candidate_name in example_names:
                continue
            optimized_files.append(opt_path)
            variant_aliases.append((candidate_name, opt_path))
        
        # Pattern 2: optimized_{name}.{ext} (e.g., optimized_moe.py / optimized_moe.cu)
        pattern2 = chapter_dir / f"optimized_{example_name}{ext}"
        if pattern2.exists():
            # Validate optimized file if requested
            if validate or warn_missing:
                opt_valid = validate_benchmark_file(pattern2, warn=warn_missing)
                if validate and not opt_valid:
                    pass  # Don't add invalid file
                else:
                    optimized_files.append(pattern2)
            else:
                optimized_files.append(pattern2)
        
        if optimized_files:
            pairs.append((baseline_file, optimized_files, example_name))
            for variant_name, opt_path in variant_aliases:
                pairs.append((baseline_file, [opt_path], variant_name))
    
    return pairs


def discover_cuda_benchmarks(repo_root: Path) -> List[Path]:
    """Discover CUDA benchmark files (files with .cu extension or in cuda/ directories).
    
    Args:
        repo_root: Path to repository root
        
    Returns:
        List of paths to CUDA benchmark files
    """
    cuda_benchmarks: List[Path] = []

    def _collect_from_dir(root: Path) -> None:
        cuda_benchmarks.extend(root.glob("*.cu"))
        cuda_subdir = root / "cuda"
        if cuda_subdir.exists() and cuda_subdir.is_dir():
            cuda_benchmarks.extend(cuda_subdir.glob("*.cu"))

    # Look for .cu files in chapter directories
    for chapter_dir in repo_root.glob("ch*/"):
        if chapter_dir.is_dir():
            _collect_from_dir(chapter_dir)

    for lab_dir in _lab_dirs(repo_root):
        _collect_from_dir(lab_dir)

    return sorted(cuda_benchmarks)


def discover_all_chapters(repo_root: Path) -> List[Path]:
    """Discover all chapter and lab directories.
    
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
    
    chapter_dirs = sorted(
        [
            d
            for d in repo_root.iterdir()
            if d.is_dir() and d.name.startswith("ch") and d.name[2:].isdigit()
        ],
        key=chapter_sort_key,
    )

    chapter_dirs.extend(list(_lab_dirs(repo_root)))
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
        try:
            normalized = normalize_chapter_token(chapter)
        except ValueError:
            chapter_dirs = []
        else:
            chapter_dir = repo_root / normalized
            chapter_dirs = [chapter_dir] if chapter_dir.exists() else []
    
    for chapter_dir in chapter_dirs:
        pairs = discover_benchmarks(chapter_dir)
        all_pairs.extend(pairs)
    
    return all_pairs
