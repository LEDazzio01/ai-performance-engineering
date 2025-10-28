#!/usr/bin/env python3
"""Blackwell-only architecture helpers for AI Performance Engineering."""

from typing import Any, Dict
import os
import torch

BLACKWELL_CC = "10.0"

class ArchitectureConfig:
    """Provide configuration details for NVIDIA Blackwell GPUs."""

    def __init__(self) -> None:
        self.arch = self._detect_architecture()
        self.config = self._get_architecture_config()

    def _detect_architecture(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        props = torch.cuda.get_device_properties(0)
        compute_capability = f"{props.major}.{props.minor}"
        return "blackwell" if compute_capability == BLACKWELL_CC else "other"

    def _get_architecture_config(self) -> Dict[str, Any]:
        blackwell = {
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
        return blackwell if self.arch == "blackwell" else generic

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
            if hasattr(cfg, "max_autotune_gemm_backends"):
                cfg.max_autotune_gemm_backends = "TRITON,CUTLASS,ATen"
        
        # Triton 3.5 configuration for Blackwell
        if self.arch == "blackwell":
            try:
                import triton
                # Configure Triton 3.5 for Blackwell (SM 10.0)
                if hasattr(triton.runtime, "driver"):
                    triton.runtime.driver.set_active_device_capability(10, 0)
            except (ImportError, AttributeError):
                pass
            
            # Blackwell-specific environment variables
            os.environ.setdefault("TRITON_CUDNN_ALGOS", "1")
            os.environ.setdefault("TRITON_TMA_ENABLE", "1")
            os.environ.setdefault("TRITON_ALWAYS_COMPILE", "0")  # Use kernel cache
        
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
        
        # PyTorch 2.9: Enable TF32 for Blackwell (improves FP32 matmul performance)
        if hasattr(torch.backends.cuda, "matmul"):
            matmul_backend = torch.backends.cuda.matmul
            if hasattr(matmul_backend, "fp32_precision"):
                matmul_backend.fp32_precision = "tf32"
            elif hasattr(matmul_backend, "allow_tf32"):
                matmul_backend.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"

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


def _install_symmetric_memory_shim() -> None:
    """Bridge PyTorch symmetric memory APIs when they are hidden under experimental modules."""
    global _SYMMETRIC_SHIM_INSTALLED
    if _SYMMETRIC_SHIM_INSTALLED:
        return

    try:
        import torch.distributed as dist
        import torch.distributed.nn  # noqa: F401 - ensures dist.nn is registered
    except ImportError:
        return

    if hasattr(dist.nn, "SymmetricMemory"):
        _SYMMETRIC_SHIM_INSTALLED = True
        return

    try:
        import torch.distributed._symmetric_memory as _symm
        import torch.distributed.distributed_c10d as c10d
        from torch._C._distributed_c10d import ProcessGroup as _ProcessGroup  # type: ignore
    except ImportError:
        return

    if not _symm.is_nvshmem_available():
        return

    class _SymmetricMemoryWrapper:
        """Minimal wrapper that mirrors torch.distributed.nn.SymmetricMemory semantics."""

        __slots__ = ("buffer", "_group", "_handle")

        def __init__(self, tensor: torch.Tensor, group=None):
            if group is None:
                group = dist.group.WORLD

            self._group = group

            try:
                backend = _symm.get_backend(tensor.device)
            except Exception:
                backend = None
            if backend != "NVSHMEM":
                try:
                    _symm.set_backend("NVSHMEM")
                except Exception:
                    pass

            self.buffer = _symm.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            try:
                if tensor.data_ptr() != self.buffer.data_ptr():
                    self.buffer.copy_(tensor)
            except RuntimeError:
                pass

            self._handle = _symm.rendezvous(self.buffer, group)

        def get_buffer(self, rank: int):
            return self._handle.get_buffer(rank)

        def barrier(self):
            dist.barrier(group=self._resolve_group())

        def _resolve_group(self):
            if isinstance(self._group, _ProcessGroup):
                return self._group
            if isinstance(self._group, str):
                return c10d._resolve_process_group(self._group)
            return dist.group.WORLD

        def __getattr__(self, name: str):
            return getattr(self._handle, name)

    dist.nn.SymmetricMemory = _SymmetricMemoryWrapper  # type: ignore[attr-defined]
    _SYMMETRIC_SHIM_INSTALLED = True


def configure_optimizations() -> None:
    global _OPTIMIZATIONS_APPLIED
    if _OPTIMIZATIONS_APPLIED:
        return
    ArchitectureConfig().configure_pytorch_optimizations()
    _install_symmetric_memory_shim()
    _OPTIMIZATIONS_APPLIED = True


arch_config = ArchitectureConfig()
configure_optimizations()
