"""Shared tcgen05 kernel loaders and Python wrappers."""

from __future__ import annotations

import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load

from common.python.tcgen05_requirements import ensure_tcgen05_supported

try:  # Ensure TORCH_CUDA_ARCH_LIST stays clamped for GB-series hosts.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional bootstrap
    arch_config = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CUTLASS_INCLUDES: list[Path] = []
# For SM100 (Blackwell) tcgen05/TMEM kernels, prefer the standalone CUTLASS which has
# the required SM100-specific headers (mma_sm100_umma.hpp, tmem_allocator_sm100.hpp).
# TransformerEngine's bundled CUTLASS may not include these newer headers.
_SM100_CUTLASS_CANDIDATES = (
    _REPO_ROOT / "third_party" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "cutlass_latest" / "cutlass-main" / "include",
)
_FALLBACK_CUTLASS_CANDIDATES = (
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
)
# Check if we have SM100+ GPU - if so, prefer the full CUTLASS
_sm100_gpu = False
try:
    major, _ = torch.cuda.get_device_capability()
    _sm100_gpu = major >= 10
except Exception:
    pass

_candidates = _SM100_CUTLASS_CANDIDATES + _FALLBACK_CUTLASS_CANDIDATES if _sm100_gpu else \
              _FALLBACK_CUTLASS_CANDIDATES + _SM100_CUTLASS_CANDIDATES
for _cand in _candidates:
    if _cand.exists():
        _CUTLASS_INCLUDES = [_cand]
        break
_CLANG_HOST = _REPO_ROOT / "third_party" / "llvm" / "bin" / "clang++"

# Build fingerprint version - bump this when changing build logic
_BUILD_FINGERPRINT_VERSION = "v2"


def _tcgen05_cuda_flags() -> list[str]:
    flags = [
        "-std=c++20",
    ]
    for inc in _CUTLASS_INCLUDES:
        flags.append(f"-I{inc}")
    # For SM100 (Blackwell), we need sm_100a to enable tcgen05/TMEM features
    # The 'a' suffix enables architecture-specific features
    major, minor = torch.cuda.get_device_capability()
    if major >= 10:
        # Use sm_100a for Blackwell to enable TMEM/tcgen05 features
        flags.append("-gencode=arch=compute_100a,code=sm_100a")
    else:
        # Fallback for older architectures
        caps: list[tuple[int, int]] = [(10, 0)]
        if major >= 12:
            caps.insert(0, (12, 0))
        elif (major, minor) not in caps:
            caps.insert(0, (major, minor))
        seen = set()
        for maj, minr in caps:
            if (maj, minr) in seen:
                continue
            seen.add((maj, minr))
            flags.append(f"-gencode=arch=compute_{maj}{minr},code=sm_{maj}{minr}")
    if _CLANG_HOST.exists():
        flags.append(f"-ccbin={_CLANG_HOST}")
    return flags


def _get_cuda_version() -> str:
    """Get CUDA toolkit version string for fingerprinting."""
    try:
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            return torch.version.cuda
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
    env_vars = ['TORCH_CUDA_ARCH_LIST', 'CUDA_HOME', 'CUDA_PATH', 'MAX_JOBS', 'CC', 'CXX']
    parts = []
    for var in sorted(env_vars):
        val = os.environ.get(var, '')
        if val:
            parts.append(f"{var}={val}")
    return "|".join(parts) if parts else "default"


def _get_include_dir_fingerprint() -> str:
    """Get fingerprint of CUTLASS include directories based on modification times."""
    hasher = hashlib.md5()
    for inc_dir in _CUTLASS_INCLUDES:
        if inc_dir.exists():
            try:
                mtime = inc_dir.stat().st_mtime
                hasher.update(f"{inc_dir}:{mtime}\n".encode())
                # Check key version files
                version_file = inc_dir / "cutlass" / "version.h"
                if version_file.exists():
                    hasher.update(f"{version_file}:{version_file.stat().st_mtime}\n".encode())
            except OSError:
                pass
    return hasher.hexdigest()[:8]


def _compute_build_fingerprint(sources: Sequence[Path], cuda_flags: list[str]) -> str:
    """Compute a hash fingerprint of all build inputs.
    
    This includes:
    - Source file contents
    - All compiler flags (including include paths)
    - Build fingerprint version (for manual invalidation)
    - Python/torch/CUDA version
    - GPU architecture
    - Environment variables
    - Include directory modification times (catches header updates)
    
    When any of these change, the cache should be invalidated.
    """
    hasher = hashlib.sha256()
    
    # Include fingerprint version for manual cache invalidation
    hasher.update(f"version:{_BUILD_FINGERPRINT_VERSION}\n".encode())
    
    # Include torch version
    hasher.update(f"torch:{torch.__version__}\n".encode())
    
    # Include CUDA version - important for toolkit upgrades
    hasher.update(f"cuda:{_get_cuda_version()}\n".encode())
    
    # Include environment variables
    hasher.update(f"env:{_get_env_fingerprint()}\n".encode())
    
    # Include GPU architecture
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            hasher.update(f"gpu_arch:sm_{major}{minor}\n".encode())
    except Exception:
        pass
    
    # Include all compiler flags (sorted for consistency)
    for flag in sorted(cuda_flags):
        hasher.update(f"flag:{flag}\n".encode())
    
    # Include source file contents
    for src in sorted(sources):
        if src.exists():
            hasher.update(f"source:{src}:\n".encode())
            hasher.update(src.read_bytes())
            hasher.update(b"\n")
    
    # Include CUTLASS include path and its fingerprint (catches header updates)
    for inc in _CUTLASS_INCLUDES:
        hasher.update(f"include:{inc}\n".encode())
    hasher.update(f"inc_fp:{_get_include_dir_fingerprint()}\n".encode())
    
    return hasher.hexdigest()[:16]  # Short hash is sufficient


def _check_and_invalidate_cache(name: str, sources: Sequence[Path], cuda_flags: list[str]) -> None:
    """Check if cached build matches current inputs; invalidate if not.
    
    This prevents stale cache issues when include paths, compiler flags,
    or source files change.
    """
    build_dir = _get_extension_build_dir(name)
    fingerprint_file = build_dir / ".build_fingerprint"
    current_fingerprint = _compute_build_fingerprint(sources, cuda_flags)
    
    # Check if we have a cached build with a matching fingerprint
    if fingerprint_file.exists():
        try:
            stored = json.loads(fingerprint_file.read_text())
            if stored.get("fingerprint") == current_fingerprint:
                return  # Cache is valid
        except (json.JSONDecodeError, KeyError):
            pass  # Invalid fingerprint file, treat as cache miss
    
    # Cache miss or fingerprint mismatch - invalidate cache
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass  # Best effort cleanup
    
    # Ensure build directory exists and write new fingerprint
    build_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_file.write_text(json.dumps({
        "fingerprint": current_fingerprint,
        "sources": [str(s) for s in sources],
        "cuda_flags": cuda_flags,
    }))


def _get_extension_build_dir(name: str) -> Path:
    """Get the torch extension build directory for a given extension name."""
    # torch extensions default to ~/.cache/torch_extensions or TORCH_EXTENSIONS_DIR
    import os
    base = os.environ.get("TORCH_EXTENSIONS_DIR")
    if base:
        return Path(base) / name
    # Fall back to workspace .torch_extensions
    return _REPO_ROOT / ".torch_extensions" / name


def _clean_stale_build(name: str) -> None:
    """Remove stale build artifacts if .so is missing but build.ninja exists."""
    import shutil
    build_dir = _get_extension_build_dir(name)
    ninja_file = build_dir / "build.ninja"
    so_file = build_dir / f"{name}.so"
    
    if ninja_file.exists() and not so_file.exists():
        # Stale build directory - ninja exists but .so missing means build failed
        try:
            shutil.rmtree(build_dir)
        except Exception:
            pass  # Best effort cleanup


def _load_extension(name: str, sources: Sequence[Path]):
    cuda_flags = _tcgen05_cuda_flags()
    
    # Check if cached build matches current inputs; invalidate if not
    # This prevents stale cache issues when include paths or flags change
    _check_and_invalidate_cache(name, sources, cuda_flags)
    
    # Clean up stale build artifacts (incomplete builds)
    _clean_stale_build(name)
    
    try:
        module = load(
            name=name,
            sources=[str(src) for src in sources],
            extra_cuda_cflags=cuda_flags,
            extra_cflags=["-std=c++20"],
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
        
        # After successful load, update fingerprint file to mark cache as valid
        build_dir = _get_extension_build_dir(name)
        fingerprint_file = build_dir / ".build_fingerprint"
        current_fingerprint = _compute_build_fingerprint(sources, cuda_flags)
        fingerprint_file.write_text(json.dumps({
            "fingerprint": current_fingerprint,
            "sources": [str(s) for s in sources],
            "cuda_flags": cuda_flags,
        }))
        
        return module
    except Exception as e:
        # On failure, retry with verbose=True to capture build errors
        error_msg = str(e)
        if "cannot open shared object file" in error_msg or "No such file" in error_msg:
            # Clean up and retry with verbose output
            _clean_stale_build(name)
            try:
                return load(
                    name=name,
                    sources=[str(src) for src in sources],
                    extra_cuda_cflags=cuda_flags,
                    extra_cflags=["-std=c++20"],
                    extra_ldflags=["-lcuda"],
                    verbose=True,  # Show build errors on retry
                )
            except Exception as retry_e:
                raise RuntimeError(
                    f"Failed to build tcgen05 extension '{name}'. "
                    f"Build errors (see above). Original error: {retry_e}"
                ) from retry_e
        raise


@lru_cache(None)
def load_matmul_tcgen05_module():
    """Compile (if needed) and return the Chapter 10 tcgen05 matmul extension."""
    import os
    from common.python.smoke import is_smoke_mode
    if is_smoke_mode():
        raise RuntimeError("SKIPPED: tcgen05 extension disabled in low-memory mode")
    return _load_extension("ch10_matmul_tcgen05_ext", [_REPO_ROOT / "ch10" / "matmul_tcgen05.cu"])


@lru_cache(None)
def load_tiling_tcgen05_module():
    """Compile (if needed) and return the Chapter 8 tcgen05 tiling extension."""
    import os
    from common.python.smoke import is_smoke_mode
    if is_smoke_mode():
        raise RuntimeError("SKIPPED: tcgen05 extension disabled in low-memory mode")
    return _load_extension("ch8_tiling_tcgen05_ext", [_REPO_ROOT / "ch8" / "tiling_kernels_tcgen05.cu"])


def matmul_tcgen05(a: torch.Tensor, b: torch.Tensor, *, module_name: str = "tcgen05 matmul") -> torch.Tensor:
    """Execute the CUTLASS tcgen05 GEMM after ensuring hardware/toolchain support."""
    ensure_tcgen05_supported(loader=load_matmul_tcgen05_module, module_name=module_name)
    module = load_matmul_tcgen05_module()
    return module.matmul_tcgen05(a, b)


def matmul_tcgen05_bias_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    *,
    module_name: str = "tcgen05 matmul bias+SiLU",
) -> torch.Tensor:
    """Execute the tcgen05 GEMM with TMEM-resident bias+SiLU epilogue."""
    ensure_tcgen05_supported(loader=load_matmul_tcgen05_module, module_name=module_name)
    module = load_matmul_tcgen05_module()
    return module.matmul_tcgen05_bias_silu(a, b, bias)


def matmul_tiling_tcgen05(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    module_name: str = "tcgen05 tiling matmul",
) -> torch.Tensor:
    """Execute the CUTLASS tcgen05 tiling GEMM."""
    ensure_tcgen05_supported(loader=load_tiling_tcgen05_module, module_name=module_name)
    module = load_tiling_tcgen05_module()
    return module.matmul_tiling_tcgen05(a, b)


__all__ = [
    "load_matmul_tcgen05_module",
    "load_tiling_tcgen05_module",
    "matmul_tcgen05",
    "matmul_tcgen05_bias_silu",
    "matmul_tiling_tcgen05",
]
