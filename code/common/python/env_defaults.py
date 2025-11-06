"""Helpers for applying environment defaults and reporting capabilities."""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Tuple

ENV_DEFAULTS: Dict[str, str] = {
    "PYTHONFAULTHANDLER": "1",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "TORCH_CUDNN_V8_API_ENABLED": "1",
    "CUDA_LAUNCH_BLOCKING": "0",
    "CUDA_CACHE_DISABLE": "0",
    "NCCL_IB_DISABLE": "0",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_SHM_DISABLE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TORCH_COMPILE_DEBUG": "0",
    # "TORCH_LOGS": "",  # Disabled - remove verbose dynamo logging to reduce noise
    "CUDA_HOME": "/usr/local/cuda-13.0",
}

CUDA_PATH_SUFFIXES: Tuple[str, ...] = ("bin",)
CUDA_LIBRARY_SUFFIXES: Tuple[str, ...] = ("lib64",)

# Try to find NCCL library for current architecture
def _find_nccl_library() -> str:
    """Find NCCL library for the current architecture."""
    import platform
    machine = platform.machine()
    
    # Try architecture-specific paths
    candidates = []
    if machine == "x86_64":
        candidates = [
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "/usr/lib/x86_64-linux-gnu/libnccl.so",
        ]
    elif machine in ("aarch64", "arm64"):
        candidates = [
            "/usr/lib/aarch64-linux-gnu/libnccl.so.2",
            "/usr/lib/aarch64-linux-gnu/libnccl.so",
        ]
    
    # Also try generic paths
    candidates.extend([
        "/usr/local/lib/libnccl.so.2",
        "/usr/local/lib/libnccl.so",
        "/usr/lib/libnccl.so.2",
        "/usr/lib/libnccl.so",
    ])
    
    # Return first existing file, or default to x86_64 path (will be ignored if not found)
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # Return empty string if not found (will be skipped when adding to LD_PRELOAD)
    return ""

NCCL_LIBRARY_PATH = _find_nccl_library()

REPORTED_ENV_KEYS: Tuple[str, ...] = (
    "PYTHONFAULTHANDLER",
    "TORCH_SHOW_CPP_STACKTRACES",
    "TORCH_CUDNN_V8_API_ENABLED",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_CACHE_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "PYTORCH_ALLOC_CONF",
    "TORCH_COMPILE_DEBUG",
    "TORCH_LOGS",
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)

_ENV_AND_CAPABILITIES_LOGGED = False


def apply_env_defaults() -> Dict[str, str]:
    """Apply default environment configuration and return the resulting values."""
    applied: Dict[str, str] = {}

    for key, value in ENV_DEFAULTS.items():
        previous = os.environ.get(key)
        if previous is None:
            os.environ.setdefault(key, value)
        applied[key] = os.environ[key]

    if "PYTORCH_ALLOC_CONF" not in os.environ:
        legacy = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["PYTORCH_ALLOC_CONF"] = legacy or "max_split_size_mb:128,expandable_segments:True"
    applied["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]

    _ensure_cuda_paths()
    applied["PATH"] = os.environ.get("PATH", "")
    applied["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")

    _ensure_ld_preload()
    applied["LD_PRELOAD"] = os.environ.get("LD_PRELOAD", "")

    return {key: os.environ.get(key, "") for key in REPORTED_ENV_KEYS}


def _ensure_cuda_paths() -> None:
    """Ensure CUDA paths are present in PATH and LD_LIBRARY_PATH."""
    cuda_home = os.environ.get("CUDA_HOME", ENV_DEFAULTS["CUDA_HOME"])

    path_prefixes = _build_paths(cuda_home, CUDA_PATH_SUFFIXES)
    lib_prefixes = _build_paths(cuda_home, CUDA_LIBRARY_SUFFIXES)

    for prefix in path_prefixes:
        _prepend_if_missing("PATH", prefix)
    for prefix in lib_prefixes:
        _prepend_if_missing("LD_LIBRARY_PATH", prefix)


def _build_paths(root: str, suffixes: Iterable[str]) -> List[str]:
    return [os.path.join(root, suffix) for suffix in suffixes]


def _prepend_if_missing(key: str, prefix: str) -> None:
    os.environ.setdefault(key, "")
    existing = os.environ.get(key, "")
    components = [segment for segment in existing.split(os.pathsep) if segment]
    if prefix not in components:
        components.insert(0, prefix)
        os.environ[key] = os.pathsep.join(components)


def _ensure_ld_preload() -> None:
    os.environ.setdefault("LD_PRELOAD", "")
    preload_entries = [segment for segment in os.environ["LD_PRELOAD"].split(os.pathsep) if segment]
    
    # Only add NCCL library if it exists and isn't already in LD_PRELOAD
    if NCCL_LIBRARY_PATH and os.path.exists(NCCL_LIBRARY_PATH):
        if NCCL_LIBRARY_PATH not in preload_entries:
            preload_entries.insert(0, NCCL_LIBRARY_PATH)
            os.environ["LD_PRELOAD"] = os.pathsep.join(preload_entries)
    elif NCCL_LIBRARY_PATH:
        # Library path was set but doesn't exist - this is okay, just skip it
        pass


def snapshot_environment() -> Dict[str, str]:
    """Return a snapshot of the relevant environment variables."""
    return {key: os.environ.get(key, "") for key in REPORTED_ENV_KEYS}


def dump_environment_and_capabilities(stream=None, *, force: bool = False) -> None:
    """Emit environment configuration and hardware capabilities."""
    global _ENV_AND_CAPABILITIES_LOGGED
    if _ENV_AND_CAPABILITIES_LOGGED and not force:
        return

    if stream is None:
        stream = sys.stdout

    env_snapshot = snapshot_environment()
    print("=" * 80, file=stream)
    print("ENVIRONMENT CONFIGURATION", file=stream)
    print("=" * 80, file=stream)
    for key in REPORTED_ENV_KEYS:
        print(f"{key}={env_snapshot.get(key, '')}", file=stream)

    print("\n" + "=" * 80, file=stream)
    print("HARDWARE CAPABILITIES", file=stream)
    print("=" * 80, file=stream)

    try:
        import torch
    except ImportError:
        print("torch not available: unable to report GPU capabilities", file=stream)
        return

    if not torch.cuda.is_available():
        print("CUDA not available on this system", file=stream)
        return

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
    except Exception as exc:
        print(f"Failed to query CUDA device: {exc}", file=stream)
        return

    print(f"GPU Name: {props.name}", file=stream)
    print(f"Compute Capability: {props.major}.{props.minor}", file=stream)
    print(f"Total Memory (GB): {props.total_memory / (1024 ** 3):.2f}", file=stream)
    print(f"SM Count: {props.multi_processor_count}", file=stream)
    print(f"CUDA Version (PyTorch): {getattr(torch.version, 'cuda', 'unknown')}", file=stream)
    cudnn_version = None
    try:
        if torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    print(f"cuDNN Version: {cudnn_version or 'unavailable'}", file=stream)
    
    # Check profiling tool availability
    print("\n" + "=" * 80, file=stream)
    print("PROFILING TOOLS", file=stream)
    print("=" * 80, file=stream)
    
    # Check nsys availability
    nsys_available = False
    try:
        import subprocess
        import shutil
        if shutil.which("nsys"):
            result = subprocess.run(
                ["nsys", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            nsys_available = result.returncode == 0
    except Exception:
        pass
    print(f"nsys available: {nsys_available}", file=stream)
    
    # Check ncu availability
    ncu_available = False
    try:
        if shutil.which("ncu"):
            result = subprocess.run(
                ["ncu", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            ncu_available = result.returncode == 0
    except Exception:
        pass
    print(f"ncu available: {ncu_available}", file=stream)
    
    _ENV_AND_CAPABILITIES_LOGGED = True
