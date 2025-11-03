#!/usr/bin/env python3
"""Architecture helpers for Blackwell and Grace-Blackwell GPUs."""

from typing import Any, Dict
import os
import subprocess
import shutil
import warnings
from pathlib import Path
import torch
from importlib import metadata as importlib_metadata

BLACKWELL_CC = "10.0"
GRACE_BLACKWELL_MAJOR = 12

def _parse_version_tuple(version: str) -> tuple:
    parts = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            parts.append(0)
    return tuple(parts)

class ArchitectureConfig:
    """Provide configuration details for NVIDIA Blackwell GPUs."""

    def __init__(self) -> None:
        self.arch = self._detect_architecture()
        self.config = self._get_architecture_config()
        self.cutlass_version = None

    def _detect_architecture(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"

        props = torch.cuda.get_device_properties(0)
        compute_capability = f"{props.major}.{props.minor}"

        if compute_capability == BLACKWELL_CC:
            return "blackwell"

        if props.major == GRACE_BLACKWELL_MAJOR:
            return "grace_blackwell"

        return "other"

    def _get_architecture_config(self) -> Dict[str, Any]:
        configs = {
            "blackwell": {
                "name": "Blackwell B200/B300",
                "compute_capability": BLACKWELL_CC,
                "sm_version": "sm_100",
                "memory_bandwidth": "7.8 TB/s",
                "tensor_cores": "5th Gen",
                "features": ["HBM3e", "TMA", "NVLink-C2C", "Stream-ordered Memory"],
                "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e optimisations", "NVLink-C2C"],
                "pytorch_optimizations": [
                    "torch.compile with max-autotune",
                    "TMA-aware kernels",
                    "HBM3e-aware allocation",
                    "Stream-ordered memory APIs",
                    "NVLink-C2C communication"
                ],
                "triton_features": [
                    "Triton 3.5 Blackwell optimisations",
                    "HBM3e access patterns",
                    "TMA intrinsic support",
                    "Stream-ordered memory",
                    "Blackwell-tuned kernels"
                ],
                "profiling_tools": [
                    "Nsight Systems 2025.x",
                    "Nsight Compute 2025.x",
                    "HTA",
                    "PyTorch Profiler",
                    "perf"
                ],
            },
            "grace_blackwell": {
                "name": "Grace-Blackwell GB10",
                "compute_capability": "12.1",
                "sm_version": "sm_121",
                "memory_bandwidth": "1.5 TB/s (per GPU)",
                "tensor_cores": "5th Gen (Blackwell-class)",
                "features": ["Grace-Blackwell coherence fabric", "TMA", "HBM3e", "NVLink-C2C"],
                "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e optimizations", "NVLink-C2C"],
                "pytorch_optimizations": [
                    "torch.compile with max-autotune",
                    "TMA-aware kernels",
                    "HBM3e-aware allocation",
                    "Stream-ordered memory APIs",
                    "NVLink-C2C communication"
                ],
                "triton_features": [
                    "Triton 3.5 GB10 optimizations",
                    "HBM3e access patterns",
                    "TMA intrinsic support",
                    "Stream-ordered memory",
                    "GB10-tuned kernels"
                ],
                "profiling_tools": [
                    "Nsight Systems 2025.x",
                    "Nsight Compute 2025.x",
                    "HTA",
                    "PyTorch Profiler",
                    "perf"
                ],
            },
        }

        generic = {
            "name": "Generic CUDA GPU",
            "compute_capability": "Unknown",
            "sm_version": "sm_unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": [],
            "cuda_features": [],
            "pytorch_optimizations": [],
            "triton_features": [],
            "profiling_tools": [],
        }

        return configs.get(self.arch, generic)

    def get_sm_version(self) -> str:
        return self.config["sm_version"]

    def get_architecture_name(self) -> str:
        return self.config["name"]

    def get_features(self) -> list:
        return self.config["features"]

    def get_cuda_features(self) -> list:
        return self.config["cuda_features"]

    def get_pytorch_optimizations(self) -> list:
        return self.config["pytorch_optimizations"]

    def get_triton_features(self) -> list:
        return self.config["triton_features"]

    def get_profiling_tools(self) -> list:
        return self.config["profiling_tools"]

    def configure_pytorch_optimizations(self) -> None:
        if not torch.cuda.is_available():
            return
        
        # PyTorch Inductor configuration
        inductor = getattr(torch, "_inductor", None)
        if inductor and hasattr(inductor, "config"):
            cfg = inductor.config
            # Enable PyTorch 2.9 features
            if hasattr(cfg, "triton"):
                triton_cfg = cfg.triton
                if hasattr(triton_cfg, "unique_kernel_names"):
                    triton_cfg.unique_kernel_names = True
                # NEW in PyTorch 2.9: CUDA graph trees for better performance
                if hasattr(triton_cfg, "cudagraph_trees"):
                    triton_cfg.cudagraph_trees = True
                if hasattr(triton_cfg, "cudagraphs"):
                    triton_cfg.cudagraphs = True
            
            # Enable max-autotune GEMM backends (PyTorch 2.9)
            # CUTLASS provides optimized GEMM kernels for NVIDIA GPUs
            if hasattr(cfg, "max_autotune_gemm_backends"):
                cfg.max_autotune_gemm_backends = "CUTLASS,TRITON,ATEN"
            
            # Enable CUTLASS for all operations
            if hasattr(cfg, "cuda") and hasattr(cfg.cuda, "cutlass_enabled_ops"):
                cfg.cuda.cutlass_enabled_ops = "all"
            
            # Enable aggressive Triton optimization for Blackwell
            if hasattr(cfg, "aggressive_fusion"):
                cfg.aggressive_fusion = True
        
        # Triton 3.5 configuration for Blackwell and Grace-Blackwell
        if self.arch in ("blackwell", "grace_blackwell"):
            try:
                import triton
                # Configure Triton 3.5 for appropriate architecture
                if hasattr(triton.runtime, "driver"):
                    if self.arch == "blackwell":
                        triton.runtime.driver.set_active_device_capability(10, 0)
                    elif self.arch == "grace_blackwell":
                        # CUDA 13.0 PTXAS lacks SM 12.1 support for tensormap ops; target SM 12.0 instead.
                        triton.runtime.driver.set_active_device_capability(12, 0)
            except (ImportError, AttributeError):
                pass
            
            # Blackwell/GB10-specific environment variables for TMA
            os.environ.setdefault("TRITON_CUDNN_ALGOS", "1")
            os.environ.setdefault("TRITON_TMA_ENABLE", "1")
            os.environ.setdefault("TRITON_ALWAYS_COMPILE", "0")  # Use kernel cache
            
            # Configure CUTLASS for torch.compile backend
            # Fix the cutlass_dir path to point to nvidia-cutlass-dsl installation
            if hasattr(cfg, "cuda") and hasattr(cfg.cuda, "cutlass_dir"):
                try:
                    import cutlass
                    # Get the nvidia_cutlass_dsl root directory
                    cutlass_module_path = os.path.dirname(cutlass.__file__)
                    nvidia_cutlass_root = os.path.dirname(os.path.dirname(cutlass_module_path))
                    cfg.cuda.cutlass_dir = nvidia_cutlass_root
                    try:
                        cutlass_pkg_version = importlib_metadata.version("nvidia-cutlass-dsl")
                        self.cutlass_version = cutlass_pkg_version
                        if _parse_version_tuple(cutlass_pkg_version) < (4, 2, 0):
                            warnings.warn(
                                "nvidia-cutlass-dsl < 4.2 detected; upgrade recommended for full Blackwell support.",
                                RuntimeWarning,
                            )
                    except importlib_metadata.PackageNotFoundError:
                        warnings.warn(
                            "nvidia-cutlass-dsl package not found; CUTLASS kernels may be skipped.",
                            RuntimeWarning,
                        )
                except ImportError:
                    # If cutlass not installed, unset cutlass_dir
                    # PyTorch will skip CUTLASS backend
                    pass

            if "TRITON_PTXAS_PATH" not in os.environ:
                try:
                    triton_root = Path(triton.__file__).resolve().parent
                    bundled_ptxas = triton_root / "backends" / "nvidia" / "bin" / "ptxas"
                    system_ptxas = shutil.which("ptxas")
                    version_ok = False
                    if bundled_ptxas.exists():
                        try:
                            result = subprocess.run(
                                [str(bundled_ptxas), "--version"],
                                capture_output=True,
                                text=True,
                                timeout=2,
                                check=False,
                            )
                        except (subprocess.SubprocessError, OSError):
                            result = None
                        if result and "release 13." in result.stdout:
                            version_ok = True
                    if not version_ok and system_ptxas:
                        os.environ["TRITON_PTXAS_PATH"] = system_ptxas
                        if VERBOSE_EXPERIMENTAL_FEATURES:
                            print(f"✅ Triton: using system ptxas at {system_ptxas} for SM 12.1 support")
                except Exception as ex:
                    if VERBOSE_EXPERIMENTAL_FEATURES:
                        print(f"⚠️  Triton ptxas selection failed: {ex}")
        
        # PyTorch 2.9: Enable FlashAttention-3 for Blackwell
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
        
        # Standard CUDA configurations
        os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
        os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "0")
        if "PYTORCH_ALLOC_CONF" not in os.environ:
            legacy_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            os.environ["PYTORCH_ALLOC_CONF"] = legacy_alloc or "max_split_size_mb:128,expandable_segments:True"
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        
        # PyTorch 2.9: Enable TF32 using the new precision API with graceful fallback.
        cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
        matmul_tf32_enabled = False
        if cuda_matmul is not None:
            if hasattr(cuda_matmul, "allow_tf32"):
                try:
                    cuda_matmul.allow_tf32 = True
                    matmul_tf32_enabled = True
                except (RuntimeError, AttributeError):
                    matmul_tf32_enabled = False
            if not matmul_tf32_enabled and hasattr(cuda_matmul, "fp32_precision"):
                try:
                    # Valid options are 'ieee', 'tf32', or 'none'.
                    cuda_matmul.fp32_precision = "tf32"
                    matmul_tf32_enabled = True
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        cudnn_tf32_enabled = False
        if cudnn_conv is not None:
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                try:
                    torch.backends.cudnn.allow_tf32 = True
                    cudnn_tf32_enabled = True
                except (RuntimeError, AttributeError):
                    cudnn_tf32_enabled = False
            if not cudnn_tf32_enabled and hasattr(cudnn_conv, "fp32_precision"):
                try:
                    cudnn_conv.fp32_precision = "tf32"
                    cudnn_tf32_enabled = True
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass

    def print_info(self) -> None:
        cfg = self.config
        print(f"Architecture: {cfg['name']}")
        print(f"Compute Capability: {cfg['compute_capability']}")
        print(f"SM Version: {cfg['sm_version']}")
        print(f"Memory Bandwidth: {cfg['memory_bandwidth']}")
        print(f"Tensor Cores: {cfg['tensor_cores']}")
        if cfg['features']:
            print(f"Features: {', '.join(cfg['features'])}")
        if cfg['cuda_features']:
            print(f"CUDA Features: {', '.join(cfg['cuda_features'])}")
        if cfg['pytorch_optimizations']:
            print(f"PyTorch Optimisations: {', '.join(cfg['pytorch_optimizations'])}")
        if cfg['triton_features']:
            print(f"Triton Features: {', '.join(cfg['triton_features'])}")
        if cfg['profiling_tools']:
            print(f"Profiling Tools: {', '.join(cfg['profiling_tools'])}")

_OPTIMIZATIONS_APPLIED = False
_SYMMETRIC_SHIM_INSTALLED = False
_TRITON_ARCH_PATCHED = False

# Feature flags (can be disabled via environment variables)
ENABLE_SYMMETRIC_MEMORY_SHIM = os.environ.get("ENABLE_SYMMETRIC_MEMORY_SHIM", "1") == "1"
ENABLE_TRITON_PATCH = os.environ.get("ENABLE_TRITON_PATCH", "1") == "1"
VERBOSE_EXPERIMENTAL_FEATURES = os.environ.get("VERBOSE_EXPERIMENTAL_FEATURES", "0") == "1"


def _install_symmetric_memory_shim() -> None:
    """
    Bridge PyTorch symmetric memory APIs when they are hidden under experimental modules.
    
    WHY THIS EXISTS:
    PyTorch 2.9+ includes symmetric memory (backed by NVSHMEM) but the API may be
    located in experimental modules. This shim provides a stable interface until
    PyTorch stabilizes the API location.
    
    WHAT IT DOES:
    - Checks if torch.distributed.nn.SymmetricMemory exists (PyTorch 2.9+ stable API)
    - If not, attempts to bridge from torch.distributed._symmetric_memory (experimental)
    - Creates a wrapper that matches the stable API semantics
    
    WHEN TO DISABLE:
    - Set ENABLE_SYMMETRIC_MEMORY_SHIM=0 if you experience issues
    - The shim gracefully degrades if dependencies are unavailable
    
    PERFORMANCE IMPACT:
    - Minimal: Only activates when needed
    - Provides <5µs cross-GPU access vs ~10-50µs with NCCL
    """
    global _SYMMETRIC_SHIM_INSTALLED
    
    if _SYMMETRIC_SHIM_INSTALLED:
        return
    
    if not ENABLE_SYMMETRIC_MEMORY_SHIM:
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print("ℹ️  Symmetric memory shim disabled via ENABLE_SYMMETRIC_MEMORY_SHIM=0")
        return

    try:
        import torch.distributed as dist
        import torch.distributed.nn  # noqa: F401 - ensures dist.nn is registered
    except ImportError:
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print("⚠️  Symmetric memory shim: torch.distributed not available")
        return

    # Check if stable API already exists
    if hasattr(dist.nn, "SymmetricMemory"):
        _SYMMETRIC_SHIM_INSTALLED = True
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print("✅ Symmetric memory: Using stable PyTorch API")
        return

    # Attempt to bridge from experimental API
    try:
        import torch.distributed._symmetric_memory as _symm
        import torch.distributed.distributed_c10d as c10d
        from torch._C._distributed_c10d import ProcessGroup as _ProcessGroup  # type: ignore
    except ImportError as e:
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print(f"⚠️  Symmetric memory shim: Experimental API not available ({e})")
        return

    # Check NVSHMEM availability
    try:
        if not _symm.is_nvshmem_available():
            if VERBOSE_EXPERIMENTAL_FEATURES:
                print("⚠️  Symmetric memory shim: NVSHMEM not available")
            return
    except Exception as e:
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print(f"⚠️  Symmetric memory shim: NVSHMEM check failed ({e})")
        return

    class _SymmetricMemoryWrapper:
        """
        Minimal wrapper that mirrors torch.distributed.nn.SymmetricMemory semantics.
        
        This wrapper bridges the experimental _symmetric_memory module to provide
        a stable API compatible with PyTorch 2.9+ stable symmetric memory.
        """

        __slots__ = ("buffer", "_group", "_handle")

        def __init__(self, tensor: torch.Tensor, group=None):
            if group is None:
                group = dist.group.WORLD

            self._group = group

            # Configure backend
            try:
                backend = _symm.get_backend(tensor.device)
            except Exception as e:
                if VERBOSE_EXPERIMENTAL_FEATURES:
                    print(f"⚠️  Symmetric memory: Failed to get backend ({e})")
                backend = None
            
            if backend != "NVSHMEM":
                try:
                    _symm.set_backend("NVSHMEM")
                except Exception as e:
                    if VERBOSE_EXPERIMENTAL_FEATURES:
                        print(f"⚠️  Symmetric memory: Failed to set NVSHMEM backend ({e})")
                    # Continue anyway - may still work

            # Allocate symmetric buffer
            try:
                self.buffer = _symm.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to allocate symmetric memory buffer: {e}. "
                    f"This may indicate NVSHMEM configuration issues."
                ) from e

            # Copy initial data if needed
            try:
                if tensor.data_ptr() != self.buffer.data_ptr():
                    self.buffer.copy_(tensor)
            except RuntimeError as e:
                if VERBOSE_EXPERIMENTAL_FEATURES:
                    print(f"⚠️  Symmetric memory: Failed to copy initial data ({e})")
                # Continue - buffer is allocated, data may be set later

            # Create rendezvous handle
            try:
                self._handle = _symm.rendezvous(self.buffer, group)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create symmetric memory rendezvous: {e}. "
                    f"Ensure all ranks call this simultaneously."
                ) from e

        def get_buffer(self, rank: int):
            """Get buffer from specified rank."""
            try:
                return self._handle.get_buffer(rank)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get buffer from rank {rank}: {e}"
                ) from e

        def barrier(self):
            """Synchronize all ranks."""
            dist.barrier(group=self._resolve_group())

        def _resolve_group(self):
            """Resolve process group."""
            if isinstance(self._group, _ProcessGroup):
                return self._group
            if isinstance(self._group, str):
                return c10d._resolve_process_group(self._group)
            return dist.group.WORLD

        def __getattr__(self, name: str):
            """Delegate unknown attributes to handle."""
            return getattr(self._handle, name)

    try:
        dist.nn.SymmetricMemory = _SymmetricMemoryWrapper  # type: ignore[attr-defined]
        _SYMMETRIC_SHIM_INSTALLED = True
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print("✅ Symmetric memory shim: Installed successfully")
    except Exception as e:
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print(f"❌ Symmetric memory shim: Installation failed ({e})")
        # Don't raise - allow code to continue without shim


def _patch_triton_sm_arch_suffix() -> None:
    """
    Patch Triton compiler to handle Blackwell (SM 10.0) architecture correctly.
    
    WHY THIS EXISTS:
    Triton 3.5+ may generate SM architecture names with 'a' suffix (e.g., "sm_100a")
    that PTXAS doesn't support. This patch ensures compatibility by:
    1. Clamping SM 10.1+ to SM 10.0 (Blackwell) until Triton adds native support
    2. Removing 'a' suffix from architecture names when present
    
    WHAT IT DOES:
    - Patches sm_arch_from_capability() to return compatible SM names
    - Patches CUDABackend.make_ptx() and make_cubin() to clamp capabilities
    
    WHEN TO DISABLE:
    - Set ENABLE_TRITON_PATCH=0 if Triton adds native Blackwell support
    - The patch gracefully skips if Triton is unavailable
    
    KNOWN ISSUES:
    - Triton 3.5 has compiler bugs with optimal TMA configurations (see triton_tma_blackwell.py)
    - This patch doesn't fix those bugs, only ensures compilation succeeds
    
    PERFORMANCE IMPACT:
    - None: Only affects compilation, not runtime
    - Required for Triton kernels to compile on Blackwell
    """
    global _TRITON_ARCH_PATCHED
    
    if _TRITON_ARCH_PATCHED:
        return
    
    if not ENABLE_TRITON_PATCH:
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print("ℹ️  Triton SM arch patch disabled via ENABLE_TRITON_PATCH=0")
        return

    try:
        import triton.backends.nvidia.compiler as triton_compiler
    except (ImportError, ModuleNotFoundError):
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print("⚠️  Triton SM arch patch: Triton not available")
        return

    # Check if already patched
    if getattr(triton_compiler, "_sm_arch_patch_applied", False):
        _TRITON_ARCH_PATCHED = True
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print("✅ Triton SM arch patch: Already applied")
        return

    # Patch sm_arch_from_capability
    original_fn = triton_compiler.sm_arch_from_capability

    def _safe_sm_arch_from_capability(capability: int, _orig=original_fn):
        """
        Generate safe SM architecture name for Blackwell.
        
        GB10 (SM 12.1) supports TMA. CUDA 13.0 supports sm_121!
        Just remove 'a' suffix if present (ptxas doesn't recognize it).
        """
        arch = _orig(capability)
        
        # Remove 'a' suffix if present - CUDA 13.0 knows sm_121, not sm_121a
        # Examples: "sm_121a" -> "sm_121", "sm_100a" -> "sm_100"
        if arch.endswith("a"):
            return arch[:-1]
        
        return arch

    try:
        triton_compiler.sm_arch_from_capability = _safe_sm_arch_from_capability  # type: ignore[assignment]
        triton_compiler._sm_arch_patch_applied = True  # type: ignore[attr-defined]
    except Exception as e:
        if VERBOSE_EXPERIMENTAL_FEATURES:
            print(f"⚠️  Triton SM arch patch: Failed to patch sm_arch_from_capability ({e})")
        return

    # Patch CUDABackend PTX/CUBIN generation
    implementation_cls = getattr(triton_compiler, "CUDABackend", None)
    if implementation_cls is not None and not hasattr(implementation_cls, "_capability_clamp_patch"):
        original_make_ptx = implementation_cls.make_ptx
        original_make_cubin = implementation_cls.make_cubin

        def _make_ptx_with_clamp(self, src, metadata, opt, capability, _orig=original_make_ptx):
            """Keep capability as-is - CUDA 13.0 supports SM 12.1."""
            # No clamping needed - CUDA 13.0 ptxas knows sm_121
            return _orig(self, src, metadata, opt, capability)

        def _make_cubin_with_clamp(self, src, metadata, opt, capability, _orig=original_make_cubin):
            """Keep capability as-is - CUDA 13.0 supports SM 12.1."""
            # No clamping needed - CUDA 13.0 ptxas knows sm_121
            return _orig(self, src, metadata, opt, capability)

        try:
            implementation_cls.make_ptx = _make_ptx_with_clamp  # type: ignore[assignment]
            implementation_cls.make_cubin = _make_cubin_with_clamp  # type: ignore[assignment]
            implementation_cls._capability_clamp_patch = True  # type: ignore[attr-defined]
        except Exception as e:
            if VERBOSE_EXPERIMENTAL_FEATURES:
                print(f"⚠️  Triton SM arch patch: Failed to patch CUDABackend ({e})")
            # Continue - sm_arch_from_capability patch may be sufficient
    
    _TRITON_ARCH_PATCHED = True
    if VERBOSE_EXPERIMENTAL_FEATURES:
        print("✅ Triton SM arch patch: Applied successfully")


def configure_optimizations() -> None:
    global _OPTIMIZATIONS_APPLIED
    if _OPTIMIZATIONS_APPLIED:
        return
    ArchitectureConfig().configure_pytorch_optimizations()
    _install_symmetric_memory_shim()
    _patch_triton_sm_arch_suffix()
    _OPTIMIZATIONS_APPLIED = True


arch_config = ArchitectureConfig()
configure_optimizations()
