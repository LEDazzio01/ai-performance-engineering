"""Template for CUDA extension loaders with hang prevention.

This template can be used by any chapter that needs to load CUDA extensions.
It includes automatic cleanup of stale build locks to prevent hangs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from types import ModuleType
import torch

try:
    from common.python.build_utils import ensure_clean_build_directory
except ImportError:
    # Fallback if build_utils not available
    def ensure_clean_build_directory(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
        """Fallback: do nothing if build_utils not available."""
        pass


# Module-level cache for loaded extensions
_EXTENSIONS: Dict[str, ModuleType] = {}


def load_cuda_extension(
    extension_name: str,
    cuda_source_file: str,
    build_dir: Optional[Path] = None,
    include_dirs: Optional[list[Path]] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
    extra_ldflags: Optional[list[str]] = None,
    verbose: bool = False,
) -> ModuleType:
    """Load a CUDA extension with automatic stale lock cleanup.
    
    Args:
        extension_name: Name of the extension (used for caching)
        cuda_source_file: Path to the .cu source file
        build_dir: Directory for build artifacts (defaults to source_dir/build)
        include_dirs: Additional include directories
        extra_cuda_cflags: Additional CUDA compiler flags
        verbose: Enable verbose compilation output
        
    Returns:
        Loaded extension module
        
    Raises:
        RuntimeError: If extension fails to load or compile
    """
    if extension_name in _EXTENSIONS:
        return _EXTENSIONS[extension_name]
    
    try:
        from torch.utils.cpp_extension import load
        
        source_path = Path(cuda_source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"CUDA source file not found: {cuda_source_file}")
        
        source_dir = source_path.parent
        
        # Default build directory
        if build_dir is None:
            build_dir = source_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean stale locks before building to prevent hangs
        ensure_clean_build_directory(build_dir)
        
        # Default include directories
        if include_dirs is None:
            include_dirs = []
        
        # Add common headers if available
        repo_root = source_path
        while repo_root.parent != repo_root:
            repo_root = repo_root.parent
            common_headers = repo_root / "common" / "headers"
            if common_headers.exists():
                include_dirs.append(common_headers)
                break
        te_cutlass = repo_root / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include"
        if te_cutlass.exists():
            include_dirs.append(te_cutlass)
        cutlass_headers = repo_root / "third_party" / "cutlass" / "include"
        if cutlass_headers.exists():
            include_dirs.append(cutlass_headers)
        
        # Default CUDA flags
        if extra_cuda_cflags is None:
            extra_cuda_cflags = ["-lineinfo", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
        
        # Add include directories to flags
        cuda_flags = extra_cuda_cflags.copy()
        for include_dir in include_dirs:
            cuda_flags.append(f"-I{include_dir}")
        
        load_kwargs = {
            "name": extension_name,
            "sources": [str(source_path)],
            "extra_cuda_cflags": cuda_flags,
            "verbose": verbose,
            "build_directory": str(build_dir),
        }
        if extra_ldflags:
            load_kwargs["extra_ldflags"] = extra_ldflags

        _EXTENSIONS[extension_name] = load(
            **load_kwargs
        )
        
        return _EXTENSIONS[extension_name]
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CUDA extension '{extension_name}': {e}\n"
            f"Source: {cuda_source_file}\n"
            f"Build dir: {build_dir}"
        ) from e
