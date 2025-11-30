"""Template for CUDA extension loaders with hang prevention.

This template can be used by any chapter that needs to load CUDA extensions.
It includes automatic cleanup of stale build locks to prevent hangs.

Features:
- Automatic cleanup of stale build locks
- Build fingerprinting to detect when include paths or flags change
- Automatic cache invalidation when build inputs change
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Optional
from types import ModuleType
import torch

try:
    from core.utils.build_utils import ensure_clean_build_directory
except ImportError:
    # Fallback if build_utils not available
    def ensure_clean_build_directory(build_dir: Path, max_lock_age_seconds: int = 300) -> None:
        """Fallback: do nothing if build_utils not available."""
        pass


# Module-level cache for loaded extensions
_EXTENSIONS: Dict[str, ModuleType] = {}

# Build fingerprint version - bump this when changing build logic
_BUILD_FINGERPRINT_VERSION = "v1"


def _get_cuda_version() -> str:
    """Get CUDA toolkit version string for fingerprinting."""
    try:
        # Try to get from torch
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            return torch.version.cuda
        # Fallback to environment
        import os
        cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', ''))
        if cuda_home:
            return cuda_home
    except Exception:
        pass
    return "unknown"


def _get_env_fingerprint() -> str:
    """Get relevant environment variables for fingerprinting."""
    import os
    env_vars = [
        'TORCH_CUDA_ARCH_LIST',
        'CUDA_HOME',
        'CUDA_PATH', 
        'MAX_JOBS',
        'CC',
        'CXX',
    ]
    parts = []
    for var in sorted(env_vars):
        val = os.environ.get(var, '')
        if val:
            parts.append(f"{var}={val}")
    return "|".join(parts) if parts else "default"


def _get_include_dir_fingerprint(include_dirs: list[Path]) -> str:
    """Get a fingerprint of include directories based on modification times.
    
    This catches cases where header files in include directories are updated
    (e.g., CUTLASS is upgraded) without having to scan all header files.
    """
    import hashlib
    hasher = hashlib.md5()
    
    for inc_dir in sorted(include_dirs):
        if inc_dir.exists():
            # Use directory mtime as a proxy for "something changed"
            try:
                mtime = inc_dir.stat().st_mtime
                hasher.update(f"{inc_dir}:{mtime}\n".encode())
                
                # Also check a few key files if they exist
                key_files = [
                    inc_dir / "cutlass" / "version.h",
                    inc_dir / "cute" / "config.hpp", 
                ]
                for kf in key_files:
                    if kf.exists():
                        hasher.update(f"{kf}:{kf.stat().st_mtime}\n".encode())
            except OSError:
                pass
    
    return hasher.hexdigest()[:8]


def cleanup_old_extension_caches(
    max_age_days: int = 7,
    dry_run: bool = False,
) -> list[Path]:
    """Clean up old, unused extension build caches.
    
    Args:
        max_age_days: Remove caches not accessed in this many days
        dry_run: If True, only report what would be deleted
        
    Returns:
        List of directories that were (or would be) deleted
    """
    import os
    import time
    
    deleted = []
    max_age_seconds = max_age_days * 24 * 60 * 60
    now = time.time()
    
    # Find extension cache directories
    cache_dirs = []
    
    # Check workspace .torch_extensions
    cwd = Path.cwd()
    repo_root = cwd
    while repo_root.parent != repo_root:
        if (repo_root / ".git").exists() or (repo_root / "core" / "common").exists():
            break
        repo_root = repo_root.parent
    
    workspace_cache = repo_root / ".torch_extensions"
    if workspace_cache.exists():
        cache_dirs.append(workspace_cache)
    
    # Check user cache
    torch_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if torch_ext_dir:
        cache_dirs.append(Path(torch_ext_dir))
    else:
        user_cache = Path.home() / ".cache" / "torch_extensions"
        if user_cache.exists():
            cache_dirs.append(user_cache)
    
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        
        for ext_dir in cache_dir.iterdir():
            if not ext_dir.is_dir():
                continue
            
            # Check last access time
            try:
                # Use the .so file's mtime if it exists, otherwise directory mtime
                so_files = list(ext_dir.glob("*.so"))
                if so_files:
                    last_used = max(f.stat().st_mtime for f in so_files)
                else:
                    last_used = ext_dir.stat().st_mtime
                
                age = now - last_used
                if age > max_age_seconds:
                    if dry_run:
                        print(f"Would delete: {ext_dir} (unused for {age/86400:.1f} days)")
                    else:
                        shutil.rmtree(ext_dir)
                    deleted.append(ext_dir)
            except OSError:
                pass
    
    return deleted


def _compute_build_fingerprint(
    source_file: Path,
    include_dirs: list[Path],
    cuda_flags: list[str],
) -> str:
    """Compute a hash fingerprint of all build inputs.
    
    This includes:
    - Source file contents
    - All include directories (and their modification times)
    - All compiler flags
    - Build fingerprint version (for manual invalidation)
    - Python/torch/CUDA version
    - Relevant environment variables
    
    When any of these change, the cache should be invalidated.
    """
    hasher = hashlib.sha256()
    
    # Include fingerprint version for manual cache invalidation
    hasher.update(f"version:{_BUILD_FINGERPRINT_VERSION}\n".encode())
    
    # Include torch version
    hasher.update(f"torch:{torch.__version__}\n".encode())
    
    # Include CUDA version - important for toolkit upgrades
    hasher.update(f"cuda:{_get_cuda_version()}\n".encode())
    
    # Include relevant environment variables
    hasher.update(f"env:{_get_env_fingerprint()}\n".encode())
    
    # Include GPU architecture if available
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            hasher.update(f"gpu_arch:sm_{major}{minor}\n".encode())
    except Exception:
        pass
    
    # Include all compiler flags (sorted for consistency)
    for flag in sorted(cuda_flags):
        hasher.update(f"flag:{flag}\n".encode())
    
    # Include all include directories (sorted for consistency)
    for inc in sorted(str(d) for d in include_dirs):
        hasher.update(f"include:{inc}\n".encode())
    
    # Include include directory fingerprint (catches header updates)
    hasher.update(f"inc_fp:{_get_include_dir_fingerprint(include_dirs)}\n".encode())
    
    # Include source file contents
    if source_file.exists():
        hasher.update(f"source:{source_file}:\n".encode())
        hasher.update(source_file.read_bytes())
        hasher.update(b"\n")
    
    return hasher.hexdigest()[:16]  # Short hash is sufficient


def _check_and_invalidate_cache(
    build_dir: Path,
    source_file: Path,
    include_dirs: list[Path],
    cuda_flags: list[str],
    verbose: bool = False,
) -> bool:
    """Check if cached build matches current inputs; invalidate if not.
    
    This prevents stale cache issues when include paths, compiler flags,
    or source files change.
    
    Returns:
        True if cache was invalidated, False if cache is valid.
    """
    fingerprint_file = build_dir / ".build_fingerprint"
    current_fingerprint = _compute_build_fingerprint(source_file, include_dirs, cuda_flags)
    
    # Check if we have a cached build with a matching fingerprint
    if fingerprint_file.exists():
        try:
            stored = json.loads(fingerprint_file.read_text())
            if stored.get("fingerprint") == current_fingerprint:
                return False  # Cache is valid
            # Cache mismatch - log why if verbose
            if verbose:
                print(f"[extension_loader] Cache invalidated for {source_file.name}")
                print(f"  Old fingerprint: {stored.get('fingerprint', 'unknown')}")
                print(f"  New fingerprint: {current_fingerprint}")
        except (json.JSONDecodeError, KeyError):
            if verbose:
                print(f"[extension_loader] Invalid fingerprint file, rebuilding {source_file.name}")
    
    # Cache miss or fingerprint mismatch - invalidate cache
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass  # Best effort cleanup
    
    # Ensure build directory exists
    build_dir.mkdir(parents=True, exist_ok=True)
    return True


def _save_build_fingerprint(
    build_dir: Path,
    source_file: Path,
    include_dirs: list[Path],
    cuda_flags: list[str],
) -> None:
    """Save the build fingerprint after a successful build."""
    fingerprint_file = build_dir / ".build_fingerprint"
    current_fingerprint = _compute_build_fingerprint(source_file, include_dirs, cuda_flags)
    fingerprint_file.write_text(json.dumps({
        "fingerprint": current_fingerprint,
        "source": str(source_file),
        "include_dirs": [str(d) for d in include_dirs],
        "cuda_flags": cuda_flags,
    }))


def _compute_multi_source_fingerprint(
    sources: list[Path],
    cuda_flags: list[str],
    cflags: list[str],
) -> str:
    """Compute fingerprint for multi-source extensions."""
    hasher = hashlib.sha256()
    hasher.update(f"version:{_BUILD_FINGERPRINT_VERSION}\n".encode())
    hasher.update(f"torch:{torch.__version__}\n".encode())
    hasher.update(f"cuda:{_get_cuda_version()}\n".encode())
    hasher.update(f"env:{_get_env_fingerprint()}\n".encode())
    
    # Include GPU architecture if available
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            hasher.update(f"gpu_arch:sm_{major}{minor}\n".encode())
    except Exception:
        pass
    
    for flag in sorted(cuda_flags + cflags):
        hasher.update(f"flag:{flag}\n".encode())
    
    # Extract include dirs from -I flags for header tracking
    include_dirs = []
    for flag in cuda_flags:
        if flag.startswith("-I"):
            include_dirs.append(Path(flag[2:]))
    if include_dirs:
        hasher.update(f"inc_fp:{_get_include_dir_fingerprint(include_dirs)}\n".encode())
    
    for src in sorted(sources):
        if src.exists():
            hasher.update(f"source:{src}:\n".encode())
            hasher.update(src.read_bytes())
            hasher.update(b"\n")
    
    return hasher.hexdigest()[:16]


def load_cuda_extension_v2(
    name: str,
    sources: list[Path],
    build_dir: Optional[Path] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
    extra_cflags: Optional[list[str]] = None,
    extra_ldflags: Optional[list[str]] = None,
    verbose: bool = False,
) -> "ModuleType":
    """Load a CUDA extension with fingerprint-based cache invalidation.
    
    This is a more flexible version that supports:
    - Multiple source files
    - Custom build directories
    - Full control over compiler flags
    
    Args:
        name: Extension name
        sources: List of source file paths
        build_dir: Build directory (auto-detected if None)
        extra_cuda_cflags: CUDA compiler flags
        extra_cflags: C++ compiler flags
        extra_ldflags: Linker flags
        verbose: Enable verbose compilation
        
    Returns:
        Loaded extension module
    """
    if name in _EXTENSIONS:
        return _EXTENSIONS[name]
    
    from torch.utils.cpp_extension import load
    
    # Resolve sources to absolute paths
    sources = [Path(s).resolve() for s in sources]
    
    # Default build directory
    if build_dir is None:
        import os
        base = os.environ.get("TORCH_EXTENSIONS_DIR")
        if base:
            build_dir = Path(base) / name
        else:
            # Use first source's parent as reference
            repo_root = sources[0].parent
            while repo_root.parent != repo_root:
                if (repo_root / ".git").exists() or (repo_root / "core" / "common").exists():
                    break
                repo_root = repo_root.parent
            build_dir = repo_root / ".torch_extensions" / name
    
    build_dir = Path(build_dir)
    
    # Default flags
    cuda_flags = list(extra_cuda_cflags) if extra_cuda_cflags else []
    cflags = list(extra_cflags) if extra_cflags else []
    ldflags = list(extra_ldflags) if extra_ldflags else []
    
    # Compute fingerprint and check cache
    fingerprint_file = build_dir / ".build_fingerprint"
    current_fp = _compute_multi_source_fingerprint(sources, cuda_flags, cflags)
    
    cache_valid = False
    if fingerprint_file.exists():
        try:
            stored = json.loads(fingerprint_file.read_text())
            cache_valid = stored.get("fingerprint") == current_fp
        except (json.JSONDecodeError, KeyError):
            pass
    
    if not cache_valid and build_dir.exists():
        # Cache invalid - clean build directory
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean stale locks
    ensure_clean_build_directory(build_dir)
    
    # Load extension
    load_kwargs = {
        "name": name,
        "sources": [str(s) for s in sources],
        "verbose": verbose,
        "build_directory": str(build_dir),
    }
    if cuda_flags:
        load_kwargs["extra_cuda_cflags"] = cuda_flags
    if cflags:
        load_kwargs["extra_cflags"] = cflags
    if ldflags:
        load_kwargs["extra_ldflags"] = ldflags
    
    try:
        _EXTENSIONS[name] = load(**load_kwargs)
        
        # Save fingerprint after successful build
        fingerprint_file.write_text(json.dumps({
            "fingerprint": current_fp,
            "sources": [str(s) for s in sources],
            "cuda_flags": cuda_flags,
            "cflags": cflags,
        }))
        
        return _EXTENSIONS[name]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CUDA extension '{name}': {e}\n"
            f"Sources: {[str(s) for s in sources]}\n"
            f"Build dir: {build_dir}"
        ) from e


def load_cuda_extension(
    extension_name: str,
    cuda_source_file: str,
    build_dir: Optional[Path] = None,
    include_dirs: Optional[list[Path]] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
    extra_ldflags: Optional[list[str]] = None,
    verbose: bool = False,
) -> ModuleType:
    """Load a CUDA extension with automatic stale lock cleanup and cache validation.
    
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
        
    Note:
        This function automatically invalidates cached builds when source files,
        include paths, or compiler flags change. This prevents stale cache issues.
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
        
        # Normalize include directories
        include_dirs = list(include_dirs) if include_dirs is not None else []

        # Collect include paths with TE CUTLASS first, then upstream CUTLASS, then everything else.
        repo_root = source_path
        while repo_root.parent != repo_root:
            repo_root = repo_root.parent
            common_headers = repo_root / "core" / "common" / "headers"
            if common_headers.exists():
                include_dirs.append(common_headers)
                break
        te_cutlass = repo_root / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include"
        cutlass_headers = repo_root / "third_party" / "cutlass" / "include"
        ordered_includes: list[Path] = []

        def _add(path: Path) -> None:
            if path.exists() and path not in ordered_includes:
                ordered_includes.append(path)

        _add(te_cutlass)
        _add(cutlass_headers)
        for inc in include_dirs:
            _add(Path(inc))
        
        # Default CUDA flags
        if extra_cuda_cflags is None:
            extra_cuda_cflags = ["-lineinfo", "--expt-relaxed-constexpr", "--expt-extended-lambda"]
        
        # Add include directories to flags
        cuda_flags = extra_cuda_cflags.copy()
        for include_dir in ordered_includes:
            cuda_flags.append(f"-I{include_dir}")
        
        # Check if cached build matches current inputs; invalidate if not
        # This prevents stale cache issues when include paths or flags change
        _check_and_invalidate_cache(build_dir, source_path, ordered_includes, cuda_flags)
        
        # Clean stale locks before building to prevent hangs
        ensure_clean_build_directory(build_dir)
        
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
        
        # Save fingerprint after successful build
        _save_build_fingerprint(build_dir, source_path, ordered_includes, cuda_flags)
        
        return _EXTENSIONS[extension_name]
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CUDA extension '{extension_name}': {e}\n"
            f"Source: {cuda_source_file}\n"
            f"Build dir: {build_dir}"
        ) from e


def _compute_inline_fingerprint(
    name: str,
    cpp_sources: str,
    cuda_sources: str,
    cuda_flags: list[str],
) -> str:
    """Compute fingerprint for inline extension sources."""
    hasher = hashlib.sha256()
    hasher.update(f"version:{_BUILD_FINGERPRINT_VERSION}\n".encode())
    hasher.update(f"torch:{torch.__version__}\n".encode())
    hasher.update(f"cuda:{_get_cuda_version()}\n".encode())
    hasher.update(f"name:{name}\n".encode())
    
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            hasher.update(f"gpu_arch:sm_{major}{minor}\n".encode())
    except Exception:
        pass
    
    for flag in sorted(cuda_flags):
        hasher.update(f"flag:{flag}\n".encode())
    
    hasher.update(f"cpp:{cpp_sources}\n".encode())
    hasher.update(f"cuda:{cuda_sources}\n".encode())
    
    return hasher.hexdigest()[:16]


def load_inline_with_fingerprint(
    name: str,
    cpp_sources: str,
    cuda_sources: str,
    extra_cuda_cflags: Optional[list[str]] = None,
    extra_include_paths: Optional[list[str]] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
) -> "ModuleType":
    """Load an inline CUDA extension with fingerprint-based cache invalidation.
    
    This wraps torch.utils.cpp_extension.load_inline with fingerprinting to detect
    when the inline source code or compiler flags change.
    
    Args:
        name: Extension name
        cpp_sources: C++ source code as string
        cuda_sources: CUDA source code as string
        extra_cuda_cflags: Additional CUDA compiler flags
        extra_include_paths: Additional include paths
        build_dir: Build directory (auto-detected if None)
        verbose: Enable verbose compilation
        
    Returns:
        Loaded extension module
    """
    if name in _EXTENSIONS:
        return _EXTENSIONS[name]
    
    from torch.utils.cpp_extension import load_inline
    import os
    
    cuda_flags = list(extra_cuda_cflags) if extra_cuda_cflags else []
    
    # Determine build directory
    if build_dir is None:
        base = os.environ.get("TORCH_EXTENSIONS_DIR")
        if base:
            build_dir = Path(base) / name
        else:
            # Find repo root
            cwd = Path.cwd()
            repo_root = cwd
            while repo_root.parent != repo_root:
                if (repo_root / ".git").exists() or (repo_root / "core" / "common").exists():
                    break
                repo_root = repo_root.parent
            build_dir = repo_root / ".torch_extensions" / name
    
    build_dir = Path(build_dir)
    
    # Compute fingerprint and check cache
    fingerprint_file = build_dir / ".build_fingerprint"
    current_fp = _compute_inline_fingerprint(name, cpp_sources, cuda_sources, cuda_flags)
    
    cache_valid = False
    if fingerprint_file.exists():
        try:
            stored = json.loads(fingerprint_file.read_text())
            cache_valid = stored.get("fingerprint") == current_fp
        except (json.JSONDecodeError, KeyError):
            pass
    
    if not cache_valid and build_dir.exists():
        # Cache invalid - clean build directory
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass
    
    build_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_build_directory(build_dir)
    
    # Build extension
    load_kwargs: dict = {
        "name": name,
        "cpp_sources": cpp_sources,
        "cuda_sources": cuda_sources,
        "verbose": verbose,
        "build_directory": str(build_dir),
    }
    if cuda_flags:
        load_kwargs["extra_cuda_cflags"] = cuda_flags
    if extra_include_paths:
        load_kwargs["extra_include_paths"] = extra_include_paths
    
    try:
        _EXTENSIONS[name] = load_inline(**load_kwargs)
        
        # Save fingerprint after successful build
        fingerprint_file.write_text(json.dumps({
            "fingerprint": current_fp,
            "cuda_flags": cuda_flags,
        }))
        
        return _EXTENSIONS[name]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load inline CUDA extension '{name}': {e}\n"
            f"Build dir: {build_dir}"
        ) from e


def invalidate_all_caches() -> list[Path]:
    """Invalidate all extension caches by deleting them.
    
    This is useful when you want to force a complete rebuild of all extensions,
    for example after upgrading CUDA or making changes to build infrastructure.
    
    Returns:
        List of directories that were deleted
    """
    import os
    
    deleted = []
    
    # Find extension cache directories
    cache_dirs = []
    
    # Check workspace .torch_extensions
    cwd = Path.cwd()
    repo_root = cwd
    while repo_root.parent != repo_root:
        if (repo_root / ".git").exists() or (repo_root / "core" / "common").exists():
            break
        repo_root = repo_root.parent
    
    workspace_cache = repo_root / ".torch_extensions"
    if workspace_cache.exists():
        cache_dirs.append(workspace_cache)
    
    # Check user cache
    torch_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if torch_ext_dir:
        cache_dirs.append(Path(torch_ext_dir))
    else:
        user_cache = Path.home() / ".cache" / "torch_extensions"
        if user_cache.exists():
            cache_dirs.append(user_cache)
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            for ext_dir in cache_dir.iterdir():
                if ext_dir.is_dir():
                    try:
                        shutil.rmtree(ext_dir)
                        deleted.append(ext_dir)
                    except OSError:
                        pass
    
    # Clear in-memory cache
    _EXTENSIONS.clear()
    
    return deleted


def get_all_extension_loaders() -> dict[str, callable]:
    """Discover all extension loader modules in the codebase.
    
    Returns:
        Dictionary mapping extension names to their loader functions
    """
    import importlib.util
    
    loaders = {}
    
    # Find repo root
    cwd = Path.cwd()
    repo_root = cwd
    while repo_root.parent != repo_root:
        if (repo_root / ".git").exists() or (repo_root / "core" / "common").exists():
            break
        repo_root = repo_root.parent
    
    # Known extension loader modules
    extension_modules = [
        ("ch06.cuda_extensions", ["coalescing", "bank_conflicts", "ilp", "launch_bounds"]),
        ("ch12.cuda_extensions", ["kernel_fusion", "graph_bandwidth", "work_queue", "cuda_graphs"]),
        ("core.common.tcgen05", ["tcgen05"]),
    ]
    
    for module_path, ext_names in extension_modules:
        try:
            mod = importlib.import_module(module_path)
            for ext_name in ext_names:
                loader_fn = getattr(mod, f"get_{ext_name}", None) or getattr(mod, f"load_{ext_name}", None)
                if loader_fn:
                    loaders[f"{module_path}.{ext_name}"] = loader_fn
        except ImportError:
            pass
    
    return loaders


def prebuild_all_extensions(verbose: bool = False) -> dict[str, tuple[bool, str]]:
    """Pre-build all known CUDA extensions.
    
    This is useful to warm up the cache before running benchmarks,
    ensuring all compilation happens upfront.
    
    Args:
        verbose: Print progress information
        
    Returns:
        Dictionary mapping extension names to (success, message) tuples
    """
    import importlib
    import sys
    
    # Find repo root and add to path
    cwd = Path.cwd()
    repo_root = cwd
    while repo_root.parent != repo_root:
        if (repo_root / ".git").exists() or (repo_root / "core" / "common").exists():
            break
        repo_root = repo_root.parent
    
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    results = {}
    
    # List of extensions to prebuild with their import paths
    extensions = [
        ("ch06.cuda_extensions", "Chapter 6 extensions"),
        ("ch12.cuda_extensions", "Chapter 12 extensions"),
        ("core.common.tcgen05", "tcgen05 (SM100+ only)"),
    ]
    
    for module_path, description in extensions:
        if verbose:
            print(f"Building {description}...", flush=True)
        
        try:
            # Import triggers build
            mod = importlib.import_module(module_path)
            results[module_path] = (True, "OK")
            if verbose:
                print(f"  ✓ {module_path}")
        except Exception as e:
            error_msg = str(e)[:100]
            results[module_path] = (False, error_msg)
            if verbose:
                print(f"  ✗ {module_path}: {error_msg}")
    
    return results


def generate_build_lockfile(output_path: Optional[Path] = None) -> dict:
    """Generate a lockfile recording the current build environment.
    
    This is useful for reproducibility - you can compare lockfiles
    to see what changed between builds.
    
    Args:
        output_path: Optional path to write the lockfile
        
    Returns:
        Dictionary containing build environment information
    """
    import platform
    import os
    import sys
    
    import datetime
    
    lockfile = {
        "generated_at": datetime.datetime.now().isoformat(),
        "fingerprint_version": _BUILD_FINGERPRINT_VERSION,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": _get_cuda_version(),
        },
        "environment": {},
        "gpu": {},
    }
    
    # Record relevant environment variables
    env_vars = ['TORCH_CUDA_ARCH_LIST', 'CUDA_HOME', 'CUDA_PATH', 'MAX_JOBS', 'CC', 'CXX']
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            lockfile["environment"][var] = val
    
    # Record GPU info
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            lockfile["gpu"]["architecture"] = f"sm_{major}{minor}"
            lockfile["gpu"]["name"] = torch.cuda.get_device_name()
            lockfile["gpu"]["count"] = torch.cuda.device_count()
    except Exception:
        pass
    
    # Record cached extensions
    lockfile["cached_extensions"] = []
    
    cwd = Path.cwd()
    repo_root = cwd
    while repo_root.parent != repo_root:
        if (repo_root / ".git").exists() or (repo_root / "core" / "common").exists():
            break
        repo_root = repo_root.parent
    
    workspace_cache = repo_root / ".torch_extensions"
    if workspace_cache.exists():
        for ext_dir in sorted(workspace_cache.iterdir()):
            if ext_dir.is_dir():
                fp_file = ext_dir / ".build_fingerprint"
                entry = {"name": ext_dir.name}
                if fp_file.exists():
                    try:
                        fp_data = json.loads(fp_file.read_text())
                        entry["fingerprint"] = fp_data.get("fingerprint", "unknown")
                    except Exception:
                        pass
                lockfile["cached_extensions"].append(entry)
    
    if output_path:
        output_path.write_text(json.dumps(lockfile, indent=2))
    
    return lockfile


if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description="CUDA Extension Cache Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old extension caches")
    cleanup_parser.add_argument("--days", type=int, default=7, help="Remove caches older than N days")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Only show what would be deleted")
    
    # invalidate command
    invalidate_parser = subparsers.add_parser("invalidate", help="Invalidate all extension caches")
    
    # info command
    info_parser = subparsers.add_parser("info", help="Show cache information")
    
    # prebuild command
    prebuild_parser = subparsers.add_parser("prebuild", help="Pre-build all CUDA extensions")
    prebuild_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # lockfile command
    lockfile_parser = subparsers.add_parser("lockfile", help="Generate build environment lockfile")
    lockfile_parser.add_argument("--output", "-o", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    if args.command == "cleanup":
        deleted = cleanup_old_extension_caches(max_age_days=args.days, dry_run=args.dry_run)
        if deleted:
            action = "Would delete" if args.dry_run else "Deleted"
            print(f"{action} {len(deleted)} old cache directories")
        else:
            print("No old caches to clean up")
    
    elif args.command == "invalidate":
        deleted = invalidate_all_caches()
        print(f"Invalidated {len(deleted)} extension caches")
        for d in deleted:
            print(f"  - {d}")
    
    elif args.command == "info":
        print(f"CUDA version: {_get_cuda_version()}")
        print(f"Torch version: {torch.__version__}")
        print(f"Environment fingerprint: {_get_env_fingerprint()}")
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                print(f"GPU architecture: sm_{major}{minor}")
        except Exception:
            pass
        print(f"Fingerprint version: {_BUILD_FINGERPRINT_VERSION}")
    
    elif args.command == "prebuild":
        print("Pre-building all CUDA extensions...")
        results = prebuild_all_extensions(verbose=True)
        
        successes = sum(1 for ok, _ in results.values() if ok)
        failures = len(results) - successes
        
        print(f"\nResults: {successes} succeeded, {failures} failed")
        if failures > 0:
            sys.exit(1)
    
    elif args.command == "lockfile":
        output_path = Path(args.output) if args.output else None
        lockfile = generate_build_lockfile(output_path)
        
        if output_path:
            print(f"Lockfile written to: {output_path}")
        else:
            print(json.dumps(lockfile, indent=2))
    
    else:
        parser.print_help()
