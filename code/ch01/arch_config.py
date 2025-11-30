# Runtime configuration helpers for Chapter 1 examples targeting NVIDIA Blackwell.
# Applies PyTorch knobs that keep us on Tensor Core fast paths and exposes a
# reusable torch.compile wrapper with safe defaults for static workloads.

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Suppress NVTX threading errors BEFORE importing torch
# This error occurs when CUDA/NVTX initializes in one thread but is used in another
# It's a known PyTorch/NVTX issue and doesn't affect functionality
# We filter at both Python sys.stderr level and file descriptor level to catch C-level writes

class FilteredStderr:
    """Filter stderr to suppress NVTX threading errors."""
    def __init__(self, original):
        self.original = original
    
    def write(self, text):
        # Filter out NVTX threading error messages
        if "External init callback must run in same thread as registerClient" not in text:
            self.original.write(text)
        # Otherwise silently drop the error message
    
    def flush(self):
        self.original.flush()
    
    def __getattr__(self, name):
        # Forward all other attributes to original stderr
        return getattr(self.original, name)

# Install stderr filter BEFORE importing torch (which triggers CUDA/NVTX initialization)
_original_stderr = sys.stderr
sys.stderr = FilteredStderr(_original_stderr)

# Also redirect at file descriptor level to catch C library writes
# Create a pipe to filter stderr at FD level
try:
    _stderr_fd = sys.stderr.fileno()
    _stderr_backup_fd = os.dup(_stderr_fd)
    _read_fd, _write_fd = os.pipe()
    
    # Redirect stderr to our pipe
    os.dup2(_write_fd, _stderr_fd)
    os.close(_write_fd)
    
    # Start a thread to filter the pipe output
    import threading
    def _filter_stderr_thread():
        """Background thread to filter stderr from pipe."""
        try:
            with open(_read_fd, 'rb', buffering=0) as pipe:
                while True:
                    data = pipe.read(4096)
                    if not data:
                        break
                    text = data.decode('utf-8', errors='replace')
                    # Filter out NVTX threading errors
                    if "External init callback must run in same thread as registerClient" not in text:
                        os.write(_stderr_backup_fd, data)
        except (OSError, ValueError):
            # Pipe closed or other error - restore and exit
            pass
        finally:
            os.close(_read_fd)
    
    _filter_thread = threading.Thread(target=_filter_stderr_thread, daemon=True)
    _filter_thread.start()
except (OSError, AttributeError):
    # If FD-level filtering fails, fall back to Python-level only
    # This can happen in some environments (e.g., Jupyter notebooks)
    pass

import torch

from core.utils.compile_utils import enable_tf32

# Suppress known warnings that don't affect functionality
# TF32 deprecation warning - PyTorch internally uses old API when set_float32_matmul_precision is called
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*", category=UserWarning)
# CUDA capability 12.1 warning - GPU is newer than officially supported but works fine
warnings.filterwarnings("ignore", message=".*Found GPU.*which is of cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)


def _configure_torch_defaults() -> None:
    """Enable TF32 Tensor Core math and cuDNN autotune."""
    enable_tf32()
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def _configure_environment() -> None:
    """Set default TorchInductor knobs that play nicely on Blackwell."""
    # Set cache directory and ensure it exists with required subdirectories
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", ".torch_inductor")
    if not os.path.isabs(cache_dir):
        # Convert relative paths to absolute paths to avoid working directory issues
        cache_dir = str(Path.cwd() / cache_dir)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    
    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        # Create subdirectories needed by PyTorch inductor for C++ compilation
        # 'od' is for output directory, 'tk' is for temporary kernel files
        (cache_path / "od").mkdir(parents=True, exist_ok=True)
        (cache_path / "tk").mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass  # If we can't create directories, PyTorch will handle it
    
    os.environ.setdefault("TORCHINDUCTOR_FUSE_TRANSPOSE", "1")
    os.environ.setdefault("TORCHINDUCTOR_FUSE_ROTARY", "1")
    os.environ.setdefault("TORCHINDUCTOR_SCHEDULING", "1")


def compile_model(module: torch.nn.Module, *, mode: str = "reduce-overhead",
                  fullgraph: bool = False, dynamic: bool = False) -> torch.nn.Module:
    """
    Compile a model with torch.compile when available.

    Defaults target steady-state inference/training loops with static shapes.
    """
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
    return compile_fn(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


_configure_environment()
_configure_torch_defaults()
